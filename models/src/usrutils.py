import torch
from torchvision.transforms import v2 as T
from .detection.engine import evaluate
from hydra.utils import instantiate
from tqdm import tqdm
import wandb
import time

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def mean_iou_torch(pred, target, num_classes, eps=1e-6):
    pred = pred.argmax(dim=1)
    target = target.squeeze() 

    pred = pred.cpu()
    target = target.cpu()

    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()

        if union == 0:
            ious.append(torch.tensor(1.0))
        else:
            ious.append(intersection / (union + eps))
    
    return torch.mean(torch.stack(ious))

def segmentation_train(cfg, train_loader, valid_loader, optimizer, scheduler, num_epochs, model, device, ishead, num_classes, issingle):
    criterion = instantiate(cfg.model.criterion)
    patience_counter = 0
    best_val_accuracy = float("-inf")
    best_state_dict = None
    for epoch in tqdm(range(num_epochs)):  # max epochs
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch['image'].to(device)
            labels = batch['mask'].to(device)
            if issingle:
                outputs = model(inputs)['out']
            else:
                outputs = model('segmentation', inputs)['out']
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss) / len(train_loader)

        model.eval()

        valid_acc = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch['image'].to(device)
                labels = batch['mask'].to(device)
                if issingle:
                    outputs = model(inputs)['out'] 
                else:
                    outputs = model('segmentation', inputs)['out']
                valid_acc = float(valid_acc + mean_iou_torch(outputs, labels, num_classes) / len(valid_loader))

        scheduler.step()

        if ishead:
            if issingle:
                wandb.log({
                    "head_train_loss": train_loss,
                    "head_val_acc": valid_acc,
                    "head_step": epoch
                })
            else:
                wandb.log({
                    "segmentation_head_train_loss": train_loss,
                    "segmentation_head_val_acc": valid_acc,
                    "head_step": epoch
                })
        else: 
            wandb.log({
                "full_train_loss": train_loss,
                "full_val_acc": valid_acc,
                "full_step": epoch
            })

        if valid_acc > best_val_accuracy:
            best_val_accuracy = valid_acc
            if issingle:
                best_state_dict = model.head.state_dict() 
            else:
                best_state_dict = model.task_heads['segmentation'].state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_state_dict

def detection_train(cfg, train_loader, valid_loader, optimizer, scheduler, num_epochs, model, device, ishead, issingle):
    patience = cfg.trainer.patience
    patience_counter = 0
    best_val_accuracy = float("-inf")
    best_state_dict = None
    for epoch in tqdm(range(num_epochs)):
        # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        #     images = list(image.to(device) for image in images)
        #     targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        train_one_epoch(model, optimizer, train_loader, device, epoch, ishead, issingle, print_freq=10)
        scheduler.step()
        coco_evaluator = evaluate(model, valid_loader, device, issingle)
        valid_acc = float(coco_evaluator.coco_eval['bbox'].stats[0])
        if ishead is True:
            wandb.log({"detection_head_acc": valid_acc, "head_step": epoch})
        else:
            wandb.log({"detection_full_acc": valid_acc, "full_step": epoch})
        if valid_acc > best_val_accuracy:
            best_val_accuracy = valid_acc
            if issingle:
                best_state_dict = model.head.state_dict()
            else:
                best_state_dict = model.task_heads['detection'].state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_state_dict

def classification_train(cfg, train_loader, valid_loader, optimizer, scheduler, num_epochs, model, device, ishead, issingle):
    criterion = instantiate(cfg.model.criterion)
    patience = cfg.trainer.patience
    patience_counter = 0
    best_val_accuracy = float("-inf")
    best_state_dict = None
    for epoch in tqdm(range(num_epochs)):  # max epochs
        model.train()
        train_loss, valid_loss = 0, 0
        total_samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if issingle:
                outputs = model(inputs)
            else:
                outputs = model('classification', inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        train_loss /= total_samples
        model.eval()
        correct, total = 0, 0
        total_samples = 0
        with torch.no_grad():
             for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if issingle:
                    outputs = model(inputs) 
                else:
                    outputs = model('classification', inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        valid_loss /= total_samples
        valid_acc = correct / total
        scheduler.step()

        if ishead:
            if issingle:
                wandb.log({
                    "head_train_loss": train_loss,
                    "head_val_acc": valid_acc,
                    "head_step": epoch
                })
            else:
                wandb.log({
                    "class_head_train_loss": train_loss,
                    "class_head_val_acc": valid_acc,
                    "head_step": epoch
                })
        else: 
            wandb.log({
                "full_train_loss": train_loss,
                "full_val_acc": valid_acc,
                "full_step": epoch
            })

        if valid_acc > best_val_accuracy:
            best_val_accuracy = valid_acc
            if issingle:
                best_state_dict = model.head.state_dict() 
            else:
                best_state_dict = model.task_heads['classification'].state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return best_state_dict

def train(cfg, dataset, optimizer, scheduler, num_epochs, model, device, num_classes):
    if cfg.model.criterion == "none":
        criterion = None
    else:
        criterion = instantiate(cfg.model.criterion)
    patience = cfg.trainer.patience
    patience_counter = 0
    best_val_accuracy = float("-inf")
    best_state_dict = None

    for epoch in tqdm(range(num_epochs)): 
        train_loss, valid_acc = train_one_epoch(cfg, dataset, model, criterion, optimizer, scheduler, device, num_classes)

        wandb.log({
            f"{cfg.model.task}_train_loss": train_loss,
            f"{cfg.model.task}_val_acc": valid_acc})

        if valid_acc > best_val_accuracy:
            best_val_accuracy = valid_acc
            best_state_dict = model.state_dict() 
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    print(f"Best validation accuracy: {best_val_accuracy}")
    return best_state_dict

def train_one_epoch(cfg, dataset, model, criterion, optimizer, scheduler, device, num_classes):
    model.train()
    train_loss = 0
    total_samples = 0
    model_type = " "
    if cfg.model.type == "single":
        for batch in dataset.train_loader:
            loss, batch_size = train_one_step(cfg, batch, model, criterion, optimizer, device, model_type)
            train_loss += loss * batch_size
            total_samples += batch_size
    else:
        pass #TODO
    train_loss /= total_samples
    valid_acc = evaluation(cfg, model, device, dataset, num_classes)
    scheduler.step()
    return train_loss, valid_acc

def train_one_step(cfg, batch, model, criterion, optimizer, device, model_type):
    model.train()
    optimizer.zero_grad()
    if (cfg.model.type == "multi" and model_type == "classification") or cfg.model.task == "classification":
        inputs, labels = batch[0].to(device), batch[1].to(device)
        if cfg.model.type == "single":
            outputs = model(inputs)
        else:
            outputs = model('classification', inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item(), inputs.size(0)       
    elif (cfg.model.type == "multi" and model_type == "detection") or cfg.model.task == "detection":
        images = batch[0]
        targets = batch[1]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        if cfg.model.type == "single":
            loss_dict = model(images, targets)
        else:
            loss_dict = model('detection', images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        loss = losses.item()
        del loss_dict, losses
        torch.cuda.empty_cache()
        return loss, len(images)
    elif (cfg.model.type == "multi" and model_type == "segmentation") or cfg.model.task == "segmentation":
        inputs, labels = batch['image'].to(device), batch['mask'].to(device)
        if cfg.model.type == "single":
            outputs = model(inputs)['out']
        else:
            outputs = model('segmentation', inputs)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item(), inputs.size(0)
    else:
        raise ValueError(f"Unsupported task: {cfg.model.task}")

def evaluation(cfg, model, device, dataset, num_classes):
    if cfg.model.type == "single":
        if cfg.model.task == "classification":
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in dataset.valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
            return correct / total
        elif cfg.model.task == "detection":
            coco_evaluator = evaluate(model, dataset.valid_loader, device, True)
            valid_acc = float(coco_evaluator.coco_eval['bbox'].stats[0])
            return valid_acc
        elif cfg.model.task == "segmentation":
            model.eval()
            valid_acc = 0
            with torch.no_grad():
                for batch in dataset.valid_loader:
                    inputs = batch['image'].to(device)
                    labels = batch['mask'].to(device)
                    outputs = model(inputs)['out'] 
                    valid_acc = float(valid_acc + mean_iou_torch(outputs, labels, num_classes) / len(dataset.valid_loader))      
            return valid_acc   
        else:
            raise ValueError(f"Unsupported task: {cfg.model.task}")
    else:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataset.classification.valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model("classification", inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0) 
        
        coco_evaluator = evaluate(model, dataset.detection.valid_loader, device, False)
        valid_acc = float(coco_evaluator.coco_eval['bbox'].stats[0])

        valid_accr = 0
        with torch.no_grad():
            for batch in dataset.segmentation.valid_loader:
                inputs = batch['image'].to(device)
                labels = batch['mask'].to(device)
                outputs = model("segmentation", inputs)['out'] 
                valid_accr = float(valid_accr + mean_iou_torch(outputs, labels, num_classes) / len(dataset.segmentation.valid_loader)) 
        
        wandb.log({
            "classification_acc": correct / total,
            "detection_acc": valid_acc,
            "segmentation_acc": valid_accr
        })

def prefixes_until(cfg, idx):
    return [f"backbone.features.{i}" for i in range(idx + 1)] if cfg.model.task == "segmentation" else [f"backbone.{i}" for i in range(idx+1)]

def measure_latency(model, input_tensor, repetitions=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(repetitions):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_latency_ms = (end_time - start_time) / repetitions * 1000
        return avg_latency_ms