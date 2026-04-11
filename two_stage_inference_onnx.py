import cv2
import numpy as np
import onnxruntime as ort
import time
import math

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
    predictions = np.squeeze(output[0], axis=0) # [num_classes + 4, num_anchors]
    predictions = predictions.T # [num_anchors, num_classes + 4]
    
    boxes = predictions[:, :4]
    scores = np.max(predictions[:, 4:], axis=1)
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return [], [], []
    
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    keep_indices = nms(boxes_xyxy, scores, iou_thres)
    
    boxes_xyxy = boxes_xyxy[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    
    boxes_xyxy[:, [0, 2]] -= pad[0]  
    boxes_xyxy[:, [1, 3]] -= pad[1]  
    boxes_xyxy[:, [0, 2]] /= ratio[0]
    boxes_xyxy[:, [1, 3]] /= ratio[1]
    
    return boxes_xyxy, scores, class_ids

def calculate_shannon_entropy(probability):
    p = max(min(probability, 1.0 - 1e-7), 1e-7)
    return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

def load_session(model_path):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 8
    sess_options.inter_op_num_threads = 8
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=[
        ('TensorrtExecutionProvider', {
            'device_id': 0, 
            'trt_fp16_enable': True, 
            'trt_engine_cache_enable': True, 
            'trt_engine_cache_path': './trt_engines',
            'trt_dump_ep_context_model': True,
            'trt_ep_context_file_path': './runs/detect'
        }), 
        ('CUDAExecutionProvider', {}),
        ('CPUExecutionProvider', {})
    ])
    return session

def get_input_info(session):
    inputs = session.get_inputs()
    shape = inputs[0].shape
    h, w = shape[2], shape[3]
    if isinstance(h, str) or h is None:
        h, w = 640, 640
    return inputs[0].name, h, w, inputs[0].type

def get_output_name(session):
    return session.get_outputs()[0].name

def preprocess(frame, input_height, input_width, input_type):
    img, ratio, pad = letterbox(frame, new_shape=(input_height, input_width))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    dtype = np.float16 if '16' in input_type else np.float32
    img = img.astype(dtype) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, ratio, pad

def main():
    ort.set_default_logger_severity(3)
    
    model1_path = "runs/detect/train16/weights/train16_best_int8.onnx"
    model2_path = "runs/detect/train17/weights/train17_best_int8.onnx"
    
    print(f"Loading Session 1: {model1_path}")
    sess1 = load_session(model1_path)
    in1_name, in1_h, in1_w, in1_type = get_input_info(sess1)
    out1_name = get_output_name(sess1)
    
    print(f"Loading Session 2: {model2_path}")
    sess2 = load_session(model2_path)
    in2_name, in2_h, in2_w, in2_type = get_input_info(sess2)
    out2_name = get_output_name(sess2)
    
    names1 = {0: 'insulator string'}
    names2 = {0: 'flashed', 1: 'broken'}
    entropy_threshold = 0.3
    
    source_path = 'footage1_aigen.mp4'
    print(f"Opening source: {source_path}")
    
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Error opening source: {source_path}")
        return
        
    target_w = 1280
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- Stage 1 Inference ---
        img1, ratio1, pad1 = preprocess(frame, in1_h, in1_w, in1_type)
        outputs1 = sess1.run([out1_name], {in1_name: img1})
        boxes1, scores1, class1_ids = postprocess(outputs1, ratio1, pad1, conf_thres=0.3, iou_thres=0.45)
        
        for i in range(len(boxes1)):
            x1, y1, x2, y2 = map(int, boxes1[i])
            conf1 = scores1[i]
            cls1_id = class1_ids[i]
            
            # Ensure within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            
            # Draw insulator
            cls1_name = names1.get(cls1_id, str(cls1_id))
            entropy = calculate_shannon_entropy(conf1)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{cls1_name} {conf1:.2f} (H:{entropy:.2f})"
            cv2.putText(frame, label_text, (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
            # Uncertainty Gating
            needs_stage_2 = True
            if cls1_id == 0 and entropy < entropy_threshold:
                needs_stage_2 = False
                
            # --- Stage 2 Inference ---
            if needs_stage_2:
                img2, ratio2, pad2 = preprocess(crop, in2_h, in2_w, in2_type)
                outputs2 = sess2.run([out2_name], {in2_name: img2})
                boxes2, scores2, class2_ids = postprocess(outputs2, ratio2, pad2, conf_thres=0.3, iou_thres=0.45)
                
                for j in range(len(boxes2)):
                    cx1, cy1, cx2, cy2 = map(int, boxes2[j])
                    conf2 = scores2[j]
                    cls2_id = class2_ids[j]
                    
                    # Convert to absolute
                    abs_x1 = x1 + cx1
                    abs_y1 = y1 + cy1
                    abs_x2 = x1 + cx2
                    abs_y2 = y1 + cy2
                    
                    cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                    cls2_name = names2.get(cls2_id, str(cls2_id))
                    cv2.putText(frame, f"{cls2_name} {conf2:.2f}", (abs_x1, max(10, abs_y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # FPS and Display
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        h, w = frame.shape[:2]
        if w > 0:
            scale = target_w / float(w)
            display_frame = cv2.resize(frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display_frame = frame
            
        cv2.imshow("ONNX Two-Stage", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
