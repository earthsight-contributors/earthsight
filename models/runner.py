import importlib
import torch
import sys
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from src.usrutils import train, prefixes_until, measure_latency

def train_single(cfg: DictConfig) -> None:

    # SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = cfg.trainer.epochs
    
    # DATASET
    dataset_module = importlib.import_module(cfg.dataset.path)
    dataset = getattr(dataset_module, cfg.dataset.cls)(cfg.trainer.batch_size)
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    num_classes = dataset.num_classes
    
    # MODEL
    model_module = importlib.import_module(cfg.model.path)
    model = getattr(model_module, cfg.model.cls)(num_classes, cfg.model.backbone)
    if cfg.model.task == "detection" or cfg.model.task == "segmentation":
        model = model.model

    if cfg.freeze == "none":
        if not all(p.requires_grad for p in model.parameters()):
            raise RuntimeError("freeze='none' but some parameters are frozen")
        prefixes = []
    elif cfg.freeze == "layer6":   
        prefixes = prefixes_until(cfg, 6)
    elif cfg.freeze == "layer7.0": 
        prefixes = prefixes_until(cfg, 6) + (["backbone.features.7.0"] if cfg.model.task == "segmentation" else ["backbone.7.0"])
    elif cfg.freeze == "layer7.1":
        assert cfg.model.backbone == "b5" or cfg.model.backbone == "b6" or cfg.model.backbone == "b7"
        prefixes = prefixes_until(cfg, 6) + (["backbone.features.7.0", "backbone.features.7.1"] if cfg.model.task == "segmentation" else ["backbone.7.0", "backbone.7.1"])
    elif cfg.freeze == "layer7.2":
        assert cfg.model.backbone == "b7"
        prefixes = prefixes_until(cfg, 6) + (["backbone.features.7.0", "backbone.features.7.1", "backbone.features.7.2"] if cfg.model.task == "segmentation" else ["backbone.7.0", "backbone.7.1", "backbone.7.2"])
    elif cfg.freeze == "layer7":   
        prefixes = prefixes_until(cfg, 7)
    elif cfg.freeze == "backbone":     
        prefixes = ["backbone."]
    else:
        raise ValueError(f"Unsupported freeze mode: {cfg.freeze}")

    param_backbone = param_head = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False
            param_backbone += (param.nelement() * param.element_size())
        else:
            param_head += (param.nelement() * param.element_size())

    buffer_backbone = buffer_head = 0
    for name, buf in model.named_buffers():
        size = buf.nelement() * buf.element_size()
        if any(name.startswith(p) for p in prefixes):
            buffer_backbone += size
        else:
            buffer_head += size

    total_mb = (param_backbone + param_head + buffer_backbone + buffer_head) / 1024**2
    backbone_mb = (param_backbone + buffer_backbone) / 1024**2
    head_mb = (param_head + buffer_head) / 1024**2

    print(f"Model size : {total_mb:.3f} MB")
    print(f"Backbone   : {backbone_mb:.3f} MB")
    print(f"Head       : {head_mb:.3f} MB")

    # for name, param in model.named_parameters():
    #     if param.requires_grad is False:
    #         print(name)
    # sys.exit(0)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = instantiate(cfg.optimizer, betas=(0.9, 0.999), params=params)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) 
    best_state_dict = train(cfg, dataset, optimizer, scheduler, num_epochs, model, device, num_classes)
    
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    latency = measure_latency(model, input_tensor, repetitions=100)
    print(f"Average Inference Latency: {latency:.3f} ms")

def multi_train(cfg, device, num_epochs, dataset, model, optimizer, scheduler): 
    criterion = instantiate(cfg.model.criterion)
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss_1 = 0
        total_samples_1 = 0
        for inputs, targets in dataset.eurosat.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, 'eurosat')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_1 += float(loss) * inputs.size(0)
            total_samples_1 += inputs.size(0)
            del inputs, targets, outputs, loss
        train_loss_1 /= total_samples_1
        wandb.log({f"{cfg.model.task}_eurosat_train_loss": train_loss_1})

        train_loss_3 = 0
        total_samples_3 = 0
        for inputs, targets in dataset.patternnet.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, 'patternnet')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_3 += float(loss) * inputs.size(0)
            total_samples_3 += inputs.size(0)
            del inputs, targets, outputs, loss
        train_loss_3 /= total_samples_3
        wandb.log({f"{cfg.model.task}_patternnet_train_loss": train_loss_3})

        train_loss_2 = 0
        total_samples_2 = 0
        for inputs, targets in dataset.advance.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, 'advance')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_2 += float(loss) * inputs.size(0)
            total_samples_2 += inputs.size(0)
            del inputs, targets, outputs, loss
        train_loss_2 /= total_samples_2
        wandb.log({f"{cfg.model.task}_advance_train_loss": train_loss_2})

        model.eval()
        with torch.no_grad():
            correct1, total1 = 0, 0
            for inputs, targets in dataset.eurosat.valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, 'eurosat')
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                correct1 += (predicted == targets).sum().item()
                total1 += targets.size(0)
                del inputs, targets, outputs, loss
            valid_acc_eurosat = correct1 / total1
        wandb.log({f"{cfg.model.task}_eurosat_val_acc": valid_acc_eurosat})

        with torch.no_grad():
            correct2, total2 = 0, 0
            for inputs, targets in dataset.advance.valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, 'advance')
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                correct2 += (predicted == targets).sum().item()
                total2 += targets.size(0)
                del inputs, targets, outputs, loss
            valid_acc_advance = correct2 / total2
        wandb.log({f"{cfg.model.task}_advance_val_acc": valid_acc_advance})

        with torch.no_grad():
            correct, total = 0, 0
            for inputs, targets in dataset.patternnet.valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, 'patternnet')
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                del inputs, targets, outputs, loss
            valid_acc_patternnet = correct / total
        wandb.log({f"{cfg.model.task}_patternnet_val_acc": valid_acc_patternnet})

        scheduler.step()

    # dl1 = dataset.eurosat.train_loader
    # dl2 = dataset.advance.train_loader
    # dl3 = dataset.patternnet.train_loader
    # eurosat = list(dl1)
    # advance = list(dl2)
    # patternnet = list(dl3)
    # min_len = min(len(eurosat), len(advance), len(patternnet))

    # for epoch in tqdm(range(num_epochs)): 
    #     model.train()
    #     train_loss = 0
    #     total_samples = 0

    #     sampled1 = random.sample(eurosat, min_len)
    #     sampled2 = random.sample(advance, min_len)
    #     sampled3 = random.sample(patternnet, min_len)

    #     zipped_loader = list(zip(sampled1, sampled2, sampled3))

    #     for (batch1, batch2, batch3) in zipped_loader:
    #         input1, label1 = batch1
    #         input2, label2 = batch2
    #         input3, label3 = batch3
    #         input1, label1 = input1.to(device), label1.to(device)
    #         input2, label2 = input2.to(device), label2.to(device)
    #         input3, label3 = input3.to(device), label3.to(device)
    #         output1 = model(input1, 'eurosat')
    #         output2 = model(input2, 'advance')
    #         output3 = model(input3, 'patternnet')
    #         loss1 = criterion(output1, label1)
    #         loss2 = criterion(output2, label2)
    #         loss3 = criterion(output3, label3)
    #         losses = [loss1, loss2, loss3]
    #         optimizer.zero_grad()
    #         optimizer.pc_backward(losses)
    #         optimizer.step()
    #         train_loss += (((float(loss1) / input1.size(0)) + (float(loss2) / input2.size(0)) + (float(loss3) / input3.size(0))) / 3)
    #         del input1, label1, input2, label2, input3, label3, output1, output2, output3
        
    #     model.eval()
        
    #     with torch.no_grad():
    #         correct, total = 0, 0
    #         for inputs, targets in dataset.eurosat.valid_loader:
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs = model(inputs, 'eurosat')
    #             _, predicted = torch.max(outputs, 1)
    #             correct += (predicted == targets).sum().item()
    #             total += targets.size(0)
    #         valid_acc_eurosat = correct / total
    #         correct, total = 0, 0
    #         for inputs, targets in dataset.advance.valid_loader:
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs = model(inputs, 'advance')
    #             _, predicted = torch.max(outputs, 1)
    #             correct += (predicted == targets).sum().item()
    #             total += targets.size(0)
    #         valid_acc_advance = correct / total
    #         correct, total = 0, 0
    #         for inputs, targets in dataset.patternnet.valid_loader:
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs = model(inputs, 'patternnet')
    #             _, predicted = torch.max(outputs, 1)
    #             correct += (predicted == targets).sum().item()
    #             total += targets.size(0)
    #         valid_acc_patternnet = correct / total
        
    #     scheduler.step()
        
    #     wandb.log({
    #         f"{cfg.model.task}_train_loss": train_loss,
    #         f"{cfg.model.task}_val_acc_eurosat": valid_acc_eurosat,
    #         f"{cfg.model.task}_val_acc_advance": valid_acc_advance,
    #         f"{cfg.model.task}_val_acc_patternnet": valid_acc_patternnet
    #     })
    
    # pattern = ["classification"] * 18 + ["segmentation"] * 6 + ["detection"] * 1
    # train_loaders = {
    #     "classification": cycle(islice(dataset.classification.train_loader, 1350)),
    #     "detection": cycle(islice(dataset.detection.train_loader, 75)),
    #     "segmentation": cycle(islice(dataset.segmentation.train_loader, 450))}
    # for param in model.backbone.parameters():
    #     param.requires_grad = False  
    
    # for epoch in tqdm(range(num_epochs)):  
    #     for i in range(75):
    #         shuffled_pattern = pattern[:] 
    #         random.shuffle(shuffled_pattern)
    #         for task in shuffled_pattern:
    #             batch = next(train_loaders[task])
    #             loss, batch_size = train_one_step(cfg, batch, model, criterion, optimizer, device, task)
    #             wandb.log({f"loss": loss})
    #         evaluation(cfg, model, device, dataset, dataset.segmentation.num_classes)
    #         scheduler.step()
    #     classification_train_loss = 0
    #     detection_train_loss = 0
    #     segmentation_train_loss = 0
    #     classification_total_samples = 0
    #     detection_total_samples = 0
    #     segmentation_total_samples = 0
    #     train_loaders = {
    #         "classification": cycle(islice(dataset.classification.train_loader, 1350)),
    #         "detection": cycle(islice(dataset.detection.train_loader, 75)),
    #         "segmentation": cycle(islice(dataset.segmentation.train_loader, 450))
    #         }
    #     for step in range(total_steps):
    #         task = next(task_cycle)
    #         batch = next(train_loaders[task])
    #         if task == "classification":
    #             inputs, labels = batch
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             optimizer.zero_grad(set_to_none=True)
    #             outputs = model('classification', inputs)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             classification_train_loss += loss.item() * inputs.size(0)
    #             classification_total_samples += inputs.size(0)
    #             del inputs, labels, outputs, loss
    #         if task == "detection":
    #             images, targets = batch
    #             images = list(image.to(device) for image in images)
    #             targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    #             losses_reduced, loss_dict_reduced, loss_value = train_one_step(issingle, model, optimizer, images, targets, scaler=None)            
    #             detection_train_loss += loss_value * len(images)
    #             detection_total_samples += len(images)
    #             del images, targets, loss_value
    #         if task == "segmentation":
    #             inputs = batch['image'].to(device)
    #             labels = batch['mask'].to(device)
    #             outputs = model('segmentation', inputs)['out']
    #             loss = criterion(outputs, labels)
    #             optimizer.zero_grad(set_to_none=True)
    #             loss.backward()
    #             optimizer.step()
    #             segmentation_train_loss += float(loss) * inputs.size(0)
    #             segmentation_total_samples += inputs.size(0)
    #             del inputs, labels, outputs, loss
    #         if step % 75 == 0:
    #             torch.cuda.empty_cache()
    #             gc.collect()
    #     classification_train_loss /= classification_total_samples
    #     detection_train_loss /= detection_total_samples
    #     segmentation_train_loss /= segmentation_total_samples
    #     wandb.log({"classification_full_train_loss": classification_train_loss, "full_step": epoch})
    #     wandb.log({"detection_full_train_loss": detection_train_loss, "full_step": epoch})
    #     wandb.log({"segmentation_full_train_loss": segmentation_train_loss, "full_step": epoch})
    #     gc.collect()
    #     torch.cuda.empty_cache()
        
    #     model.eval()
    #     correct, total = 0, 0
    #     with torch.no_grad():
    #          for inputs, targets in dataset.classification.valid_loader:
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs = model('classification', inputs)
    #             loss = criterion(outputs, targets)
    #             _, predicted = torch.max(outputs, 1)
    #             correct += (predicted == targets).sum().item()
    #             total += targets.size(0)
    #             del inputs, targets, outputs, loss
    #     valid_acc = correct / total      
    #     wandb.log({"classification_full_val_acc": valid_acc, "full_step": epoch})
    #     gc.collect()
    #     torch.cuda.empty_cache()     
        
    #     with torch.no_grad():
    #         coco_evaluator = evaluate(model, dataset.detection.valid_loader, device, issingle)
    #         valid_acc = float(coco_evaluator.coco_eval['bbox'].stats[0])
    #         wandb.log({"detection_full_val_acc": valid_acc, "full_step": epoch})
    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     with torch.no_grad():
    #         valid_acc = 0
    #         for batch in dataset.segmentation.valid_loader:
    #             inputs = batch['image'].to(device)
    #             labels = batch['mask'].to(device)
    #             outputs = model('segmentation', inputs)['out']
    #             valid_acc += float(mean_iou_torch(outputs, labels, dataset.segmentation.num_classes)) / len(dataset.segmentation.valid_loader)
    #             del inputs, labels, outputs
    #     wandb.log({"segmentation_full_val_acc": valid_acc, "full_step": epoch})
    #     gc.collect()
    #     torch.cuda.empty_cache()

    scheduler.step()

        # elif task == "detection":
        #     images, targets = batch
        #     images = list(image.to(device) for image in images)
        #     targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        #     losses_reduced, loss_dict_reduced, loss_value = train_one_step(issingle, model, optimizer, images, targets, scaler=None)
        #     train_loss += loss_value * len(images)
        #     total_samples += len(images)
        # elif task == "segmentation":
        #     inputs = batch['image'].to(device)
        #     targets = batch['mask'].to(device)
    # for epoch in tqdm(range(num_epochs)):
    #     model.train()
    #     train_one_epoch(model, optimizer, dataset.train_loader, device, epoch, ishead, print_freq=10)
    #     scheduler.step()
    #     coco_evaluator = evaluate(model, dataset.valid_loader, device=device)
    #     valid_acc = float(coco_evaluator.coco_eval['bbox'].stats[0])
    #     if ishead is True:
    #         wandb.log({"head_acc": valid_acc, "head_step": epoch})
    #     else:
    #         wandb.log({"full_acc": valid_acc, "full_step": epoch})
    #     if valid_acc > best_val_accuracy:
    #         best_val_accuracy = valid_acc
    #         best_state_dict = model.head.state_dict()
    #         # best_state_dict = model.task_heads['detection'].state_dict()
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= patience:
    #             break
    # return best_state_dict

def train_multi(cfg: DictConfig) -> None:

    # SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = cfg.trainer.epochs
    best_state_dict = None

    # DATASET
    dataset_module = importlib.import_module(cfg.dataset.path)
    dataset = getattr(dataset_module, cfg.dataset.cls)(cfg.trainer.batch_size)
    
    # MODEL
    model_module = importlib.import_module(cfg.model.path)
    if cfg.model.name == "hps_classification":
        num_classes_dict = { "eurosat" : dataset.eurosat.num_classes, "advance": dataset.advance.num_classes, "patternnet": dataset.patternnet.num_classes}
    elif cfg.model.name == "hps_segmentation":
        num_classes_dict = { "deepglobelandcover" : dataset.deepglobelandcover.num_classes, "landcover_ai": dataset.landcover_ai.num_classes, "loveda": dataset.loveda.num_classes}
    elif cfg.model.name == "multitask_model":
        num_classes_dict = {"eurosat" : dataset.eurosat.num_classes, "landcover_ai": dataset.landcover_ai.num_classes}
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")
    
    model = getattr(model_module, cfg.model.cls)(num_classes_dict)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = instantiate(cfg.optimizer, betas=(0.9, 0.999), params=params)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) 
    
    multi_train(cfg, device, num_epochs, dataset, model, optimizer, scheduler)
    
    # input_tensor = torch.randn(1, 3, 224, 224).to(device)
    # latency = measure_latency(model, input_tensor, repetitions=100)
    # print(f"Average Inference Latency: {latency:.3f} ms")

def run_experiment(cfg: DictConfig) -> None:
    if cfg.model.type == "single":
        train_single(cfg)
    else:
        train_multi(cfg)
