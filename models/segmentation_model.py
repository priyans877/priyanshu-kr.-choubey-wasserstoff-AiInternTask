import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import cv2

def segment_image(image, confidence_threshold=0.7, overlap_threshold=0.3):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()

    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    masks = prediction['masks'].squeeze().cpu().numpy()
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()

    #filter out low-confidence prediction
    high_conf_indices = np.where(scores > confidence_threshold)[0]
    masks = masks[high_conf_indices]
    boxes = boxes[high_conf_indices]
    scores = scores[high_conf_indices]
    labels = labels[high_conf_indices]

    #ramoving overlapping boxes
    keep = non_max_suppression(boxes, scores, overlap_threshold)
    masks = masks[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    segmented_objects = []
    for i, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
        mask = mask > 0.5
        box = box.astype(int)
        obj_image = image.crop(box)
        
        segmented_objects.append({
            'id': f'obj_{i+1}',
            'image': obj_image,
            'mask': mask,
            'box': box.tolist(),
            'score': float(score),
            'label': int(label)
        })

    visualization = visualize_segmentation(image, masks, boxes)

    return {
        'objects': segmented_objects,
        'visualization': visualization
    }

def non_max_suppression(boxes, scores, overlap_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]

    return keep

def visualize_segmentation(image, masks, boxes):
    image_np = np.array(image)
    for mask, box in zip(masks, boxes):
        mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)
        cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    return Image.fromarray(image_np)
