from cProfile import label
import math
from readline import read_init_file
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from PIL import Image
from myVisualize import display_instances


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
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

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

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
def evaluate(model, data_loader,img_write, device):
    #n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    #torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    for images, targets in metric_logger.log_every(data_loader, 100, header):

        images = list(img.to(device) for img in images)
   
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        # Visualization================ for 2 classes: [bg, module]
        if img_write:
            m_batch_size=len(list(res.keys()))
            for i in range (m_batch_size):
                my_image_id=list(res.keys())[i]
                my_boxes=res[my_image_id]['boxes']
                my_masks=(res[my_image_id]['masks']).squeeze(dim=1)
                class_ids=res[my_image_id]['labels']
                class_names=["bg","module","bathPod"]
                my_masks=torch.permute(my_masks,(1,2,0))
                myBoxes = torch.cat((my_boxes, class_ids.unsqueeze(dim=1)), 1)
                my_image=torch.permute(images[i],(1,2,0))

                display_instances(my_image, myBoxes, my_masks, class_ids, class_names,
                                    scores=None, title="",
                                figsize=(16, 16), ax=None,
                                show_mask=True, show_bbox=True,
                                colors=None, captions=None)
        #================================
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # mAP = coco_evaluator.coco_eval['bbox'].stats[0]
    # print(coco_evaluator.coco_eval.keys())
    # print(mAP)
    # exit()
    #TODO: Visualize the mAP for each epoch 
    # Plot the mAP
    # class_names = { "Module":1,"BathPod":2}
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.bar(class_names, mAP)
    # plt.xticks(rotation=90)
    # plt.ylim([0, 1])
    # plt.ylabel("mAP")
    # plt.show()
    #torch.set_num_threads(n_threads)
    return coco_evaluator
