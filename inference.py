from ultralytics import YOLO
import cv2
from datetime import datetime
import json
from pathlib import Path


def run_predict(model_path, source_path, show_results=False, delay=3000):
    model = YOLO(model_path)

    run_timestamp = datetime.now().isoformat()

    results = model.predict(
        source_path,
        stream=True,
        save=True,
        save_frames=True,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        line_width=3,
    )

    log_data = []
    save_dir = None

    try:
        for r in results:
            save_dir = r.save_dir

            if show_results:
                img = r.plot(line_width=3)
                h, w = img.shape[:2]
                target_w = 1280
                if w > 0:
                    scale = target_w / float(w)
                    img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

                cv2.imshow("Result", img)

                # For video/stream sources, keep UI responsive and allow quitting.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            confidences = []
            if getattr(r, "boxes", None) is not None and getattr(r.boxes, "conf", None) is not None:
                confidences = r.boxes.conf.detach().cpu().tolist() if hasattr(r.boxes.conf, "detach") else r.boxes.conf.tolist()

            log_entry = {
                "run_timestamp": run_timestamp,
                "frame_timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "source": getattr(r, "path", source_path),
                "classes": getattr(r, "names", {}),
                "speed_ms": getattr(r, "speed", {}),
                "detection": {
                    "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
                    "summary": r.summary(normalize=True),
                },
            }
            log_data.append(log_entry)

        if save_dir is not None:
            log_file = Path(save_dir) / "log.json"
            log_file.write_text(json.dumps(log_data, indent=4), encoding="utf-8")
            print(f"[/] Inference log saved to {log_file}")
        else:
            print("[!] No result")

        # If you only want a fixed delay for still images, do it here.
        if show_results and delay and delay > 0:
            cv2.waitKey(delay)

    finally:
        if show_results:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    models = [
        "optimised_yolo11/final_training_run/weights/best.pt"
    ]
    source = "footage1_aigen.mp4"
    # source = "datasets/idid-v3/images/test/170797.JPG"

    for model in models:
        print(f"\n[/] Running inference using model: {model}")
        print(f"[/] Source: {source}")

        run_predict(
            model_path=model,
            source_path=source,
            show_results=True,
            delay=0
        )
