import cv2
import numpy as np
import time
from ultralytics import YOLO

def main():
    # 1. Load models
    model1_path = 'runs/detect/train16/weights/best.pt' # Insulator detection
    model2_path = 'runs/detect/train17/weights/best.pt' # Defect detection (flashed/broken)
    
    print(f"Loading Model 1: {model1_path}")
    model1 = YOLO(model1_path)
    
    print(f"Loading Model 2: {model2_path}")
    model2 = YOLO(model2_path)

    source_path = 'footage1_aigen.mp4'
    # source_path = 'datasets/idid-v3/images/test/170797.JPG'
    print(f"Opening source: {source_path}")
    
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        # Fallback to image if video cannot be opened
        print("Image")
        frame = cv2.imread(source_path)
        if frame is None:
             print(f"Error opening source: {source_path}")
             return
        process_and_show_frame(frame, model1, model2)
        cv2.waitKey(0)
    else:
        # Process Video stream
        print("Video")
        target_w = 1280
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = process_frame(frame, model1, model2)
            
            # Display logic
            h, w = processed_frame.shape[:2]
            if w > 0:
                scale = target_w / float(w)
                display_frame = cv2.resize(processed_frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                display_frame = processed_frame
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Two-Stage Inference", display_frame)
            
            # Keep UI responsive and allow quitting
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        cap.release()
        
    cv2.destroyAllWindows()


def process_frame(frame, model1, model2):
    # 2. Run model 1 to detect insulators
    results1 = model1(frame, verbose=False)
    
    for result in results1:
        boxes = result.boxes
        if boxes:
            for box in boxes:
                # Get xyxy coordinates for the cropped region
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                # Crop the insulator from the source image
                crop = frame[y1:y2, x1:x2]
                
                # Draw insulator box (Green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf1 = box.conf[0].item()
                cls1_name = model1.names[int(box.cls[0].item())]
                cv2.putText(frame, f"{cls1_name} {conf1:.2f}", (x1, max(10, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 3. Pass cropped image to model 2 to detect flashed/broken classes
                results2 = model2(crop, verbose=False)
                
                for r2 in results2:
                    d_boxes = r2.boxes
                    if d_boxes:
                        for d_box in d_boxes:
                            # Get cropped relative coordinates
                            cx1, cy1, cx2, cy2 = map(int, d_box.xyxy[0].tolist())
                            
                            # Convert to absolute bounding boxes on the original frame
                            abs_x1 = x1 + cx1
                            abs_y1 = y1 + cy1
                            abs_x2 = x1 + cx2
                            abs_y2 = y1 + cy2
                            
                            # Draw defect box (Red)
                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                            conf2 = d_box.conf[0].item()
                            cls2_name = model2.names[int(d_box.cls[0].item())]
                            cv2.putText(frame, f"{cls2_name} {conf2:.2f}", (abs_x1, max(10, abs_y1 - 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        
    return frame

def process_and_show_frame(frame, model1, model2):
     processed_frame = process_frame(frame, model1, model2)
     target_w = 1280
     h, w = processed_frame.shape[:2]
     if w > 0:
         scale = target_w / float(w)
         display_frame = cv2.resize(processed_frame, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
     else:
         display_frame = processed_frame
     cv2.imshow("Two-Stage Inference", display_frame)

if __name__ == "__main__":
    main()
