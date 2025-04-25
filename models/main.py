import hydra
from omegaconf import DictConfig, OmegaConf
from runner import run_experiment
import wandb
from util_simulator import load_mt_model, load_dataloaders
import torch
import os
import sys
from tqdm import tqdm
from src.usrutils import mean_iou_torch

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:

    # Set checkpoint directory path
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'checkpoints')

    # Load model and dataloaders
    model = load_mt_model()
    dataloaders = load_dataloaders(batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained weights for each task
    parameters_classification = torch.load(os.path.join(checkpoint_path, 'classification_head.pth'))
    parameters_segmentation = torch.load(os.path.join(checkpoint_path, 'segmentation_head.pth'))

    # Move model to the proper device and set it to eval mode
    model.to(device)
    model.eval()

    # --- Classification Evaluation (EuroSAT) ---

    # IMPORTANT:
    # You MUST call load_state_dict with the classification weights
    # *EVERY TIME* when you switch to perform classification inference.
    # This is NOT a one-time setup.
    model.load_state_dict(parameters_classification, strict=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloaders['classification_val'], desc="Evaluating Classification"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, 'classification')
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    classification_acc = correct / total
    print(f"Classification Accuracy (EuroSAT): {classification_acc:.4f}")

    # --- Segmentation Evaluation (LandCover.ai) ---

    # IMPORTANT:
    # You MUST call load_state_dict with the segmentation weights
    # *EVERY TIME* when you switch to perform segmentation inference.
    # This is NOT a one-time setup.
    model.load_state_dict(parameters_segmentation, strict=False)

    iou_total = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloaders['segmentation_val'], desc="Evaluating Segmentation"):
            inputs = batch['image'].to(device)
            labels = batch['mask'].to(device)
            outputs = model(inputs, 'segmentation')
            iou = mean_iou_torch(outputs, labels, num_classes=5)
            iou_total += iou.item()

    segmentation_mIoU = iou_total / len(dataloaders['segmentation_val'])
    print(f"Segmentation mIoU (LandCover.ai): {segmentation_mIoU:.4f}")

    # wandb.init(
    #     project = cfg.project.name,
    #     name = cfg.model.type + '_' + cfg.freeze + '_' + cfg.model.backbone + '_' + cfg.model.name + '_' + cfg.dataset.name,
    #     config = OmegaConf.to_container(cfg, resolve=True) 
    # )
    
    # run_experiment(cfg)
    
if __name__ == "__main__":
    main()