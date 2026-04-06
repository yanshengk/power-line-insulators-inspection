import cv2
import numpy as np
import onnxruntime as ort
import time

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def compute_iou(box, boxes):
    # Compute IOU between box and boxes.
    # box: [xmin, ymin, xmax, ymax]
    # boxes: [N, 4]
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    
    return intersection_area / np.maximum(union_area, 1e-6)

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes

def postprocess(output, ratio, pad, conf_thres=0.3, iou_thres=0.45):
    # YOLO output shape is typically [1, num_classes + 4, num_anchors]
    predictions = np.squeeze(output[0], axis=0) # [num_classes + 4, num_anchors]
    predictions = predictions.T # [num_anchors, num_classes + 4]
    
    boxes = predictions[:, :4]
    scores = np.max(predictions[:, 4:], axis=1)
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    # Filter by confidence
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return [], [], []
    
    # Convert cx, cy, w, h to xmin, ymin, xmax, ymax
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    # Perform NMS
    keep_indices = nms(boxes_xyxy, scores, iou_thres)
    
    boxes_xyxy = boxes_xyxy[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    
    # Scale boxes back to original image
    boxes_xyxy[:, [0, 2]] -= pad[0]  # x padding
    boxes_xyxy[:, [1, 3]] -= pad[1]  # y padding
    boxes_xyxy[:, [0, 2]] /= ratio[0]
    boxes_xyxy[:, [1, 3]] /= ratio[1]
    
    return boxes_xyxy, scores, class_ids

if __name__ == "__main__":
    model_path = "optimised_yolo11/final_training_run/weights/best.onnx"
    video_path = "footage1_aigen.mp4"
    
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape
    input_height, input_width = 1280, 1280
        
    model_outputs = session.get_outputs()
    output_name = model_outputs[0].name
    
    print(f"Model Input: {input_name} {input_shape}")
    print(f"Model Output: {output_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        exit()
        
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        orig_img = frame.copy()
        
        # Preprocess
        img, ratio, pad = letterbox(frame, new_shape=(input_height, input_width))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Inference
        outputs = session.run([output_name], {input_name: img})
        
        # Postprocess
        boxes, scores, class_ids = postprocess(outputs, ratio, pad, conf_thres=0.3, iou_thres=0.45)
        
        # Draw bounding boxes
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            class_id = class_ids[i]
            score = scores[i]
            
            # Draw rectangle
            cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Put text
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(orig_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # FPS and Display
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(orig_img, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        target_w = 1920
        h, w = orig_img.shape[:2]
        if w > 0:
            scale = target_w / float(w)
            display_frame = cv2.resize(orig_img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display_frame = orig_img
            
        cv2.imshow("ONNX Inference", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total time elapsed: {time.time() - start_time:.2f}s for {frame_count} frames.")
