import math
import sys
import time
import wandb
import torch
import torchvision.models.detection.mask_rcnn
from . import utils
from src.detection.coco_eval import CocoEvaluator
from src.detection.coco_utils import get_coco_api_from_dataset
import gc

def train_one_step(issingle, model, optimizer, images, targets, scaler=None):
    with torch.amp.autocast('cuda', enabled=scaler is not None):
        if issingle:
            loss_dict = model(images, targets)
        else:
            loss_dict = model('detection', images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    loss_value = losses_reduced.item()

    if not math.isfinite(loss_value):
        print(f"Loss is {loss_value}, stopping training")
        print(loss_dict_reduced)
        sys.exit(1)

    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses.backward()
        optimizer.step()
    return losses_reduced, loss_dict_reduced, loss_value

def train_one_epoch(model, optimizer, data_loader, device, epoch, ishead, issingle, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    train_loss = 0
    total_samples = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        losses_reduced, loss_dict_reduced, loss_value = train_one_step(issingle, model, optimizer, images, targets, scaler)
        # with torch.amp.autocast('cuda', enabled=scaler is not None):
        #     if issingle:
        #         loss_dict = model(images, targets)
        #     else:
        #         loss_dict = model('detection', images, targets)
        #     losses = sum(loss for loss in loss_dict.values())

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # loss_value = losses_reduced.item()

        # if not math.isfinite(loss_value):
        #     print(f"Loss is {loss_value}, stopping training")
        #     print(loss_dict_reduced)
        #     sys.exit(1)

        # optimizer.zero_grad()
        # if scaler is not None:
        #     scaler.scale(losses).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     losses.backward()
        #     optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        train_loss += loss_value * len(images)
        total_samples += len(images)

    train_loss /= total_samples 
    if ishead:
        if issingle:
            wandb.log({"head_train_loss": train_loss, "head_step": epoch})
        else:
            wandb.log({"detection_head_train_loss": train_loss, "head_step": epoch})
    else:
        wandb.log({"full_train_loss": train_loss, "full_step": epoch})

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, issingle):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            if issingle:
                outputs = model(images)
            else:
                outputs = model('detection', images) 

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
            del images, outputs, res
            torch.cuda.empty_cache() 

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    return coco_evaluator