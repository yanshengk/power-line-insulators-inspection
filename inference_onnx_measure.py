"""
inference_onnx_measure.py  —  ONNX Baseline Inference with Data Collection
===================================================================
Runs the single-model YOLO11n ONNX baseline on a video source and
collects per-frame latency and power data for thesis evaluation.

Measurement methodology:
  - Wall-clock latency: time from cap.read() to end of postprocess,
    covering the full pipeline cost visible to a downstream consumer.
  - Power (VDD_IN): read from jtop at the end of each measured frame.
  - First WARMUP_FRAMES frames are excluded from all statistics because
    TensorRT kernel autotuning on real inputs produces anomalously high
    latency that is not representative of steady-state performance.
  - Results are saved to a JSON file for later analysis.
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import json
import numpy as np

# jtop is the Jetson-specific library for reading hardware telemetry.
# If it is not installed: pip install jetson-stats
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    print("[!] jtop not found. Power measurements will not be recorded.")
    print("    Install with: pip install jetson-stats")
    JTOP_AVAILABLE = False

# Number of real-input frames to discard at the start of the run.
# These are excluded because TensorRT performs kernel autotuning on the
# first few unique input shapes, inflating their latency significantly.
WARMUP_FRAMES = 30


# ── Preprocessing ──────────────────────────────────────────────────────────────

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=False, scaleFill=False, scaleup=True):
    """Resize and pad an image to new_shape while preserving aspect ratio."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def preprocess(frame, input_height, input_width, input_type):
    """Convert a BGR frame to a normalised, channel-first float tensor."""
    img, ratio, pad = letterbox(frame, new_shape=(input_height, input_width))
    img = img[:, :, ::-1].transpose(2, 0, 1)   # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    dtype = np.float16 if '16' in str(input_type) else np.float32
    img = img.astype(dtype) / 255.0
    img = np.expand_dims(img, axis=0)           # add batch dimension
    return img, ratio, pad


# ── Postprocessing ─────────────────────────────────────────────────────────────

def compute_iou(box, boxes):
    """Compute IoU between a single box and an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area   = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return intersection / np.maximum(union, 1e-6)


def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression; returns indices of kept boxes."""
    sorted_indices = np.argsort(scores)[::-1]
    keep = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep.append(box_id)
        ious = compute_iou(boxes[box_id], boxes[sorted_indices[1:]])
        keep_mask = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_mask + 1]
    return keep


def postprocess(output, ratio, pad, conf_thres=0.3, iou_thres=0.45):
    """
    Decode YOLO output, apply confidence filtering and NMS, and map
    boxes back to original image coordinates.
    """
    predictions = np.squeeze(output[0], axis=0).T   # [N, 4 + num_classes]

    boxes    = predictions[:, :4]
    scores   = np.max(predictions[:, 4:], axis=1)
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    mask   = scores > conf_thres
    boxes  = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], [], []

    # Convert centre-format (cx, cy, w, h) to corner-format (x1, y1, x2, y2)
    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, scores, iou_thres)
    boxes_xyxy = boxes_xyxy[keep]
    scores     = scores[keep]
    class_ids  = class_ids[keep]

    # Remove letterbox padding and rescale to original image dimensions
    boxes_xyxy[:, [0, 2]] -= pad[0]
    boxes_xyxy[:, [1, 3]] -= pad[1]
    boxes_xyxy[:, [0, 2]] /= ratio[0]
    boxes_xyxy[:, [1, 3]] /= ratio[1]

    return boxes_xyxy, scores, class_ids


# ── Session helpers ────────────────────────────────────────────────────────────

def load_session(model_path, trt_ep_context_file_path='./trt_engines'):
    """
    Load an ONNX model and configure ONNX Runtime to use the TensorRT
    Execution Provider with FP16 enabled and engine caching on disk.
    The fallback chain is TensorRT → CUDA → CPU, so if TensorRT is
    unavailable the session will still run (but more slowly).
    """
    sess_options = ort.SessionOptions()
    # These thread counts are kept identical to the two-stage script so
    # that thread scheduling cannot explain any throughput difference.
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=[
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_engines',
                'trt_dump_ep_context_model': True,
                'trt_ep_context_file_path': trt_ep_context_file_path,
            }),
            ('CUDAExecutionProvider', {}),
            ('CPUExecutionProvider', {}),
        ]
    )
    return session


def get_input_info(session, default_h=640, default_w=640):
    """Return the input name, spatial dimensions, and data type."""
    inputs = session.get_inputs()
    shape  = inputs[0].shape
    h, w   = shape[2], shape[3]
    if isinstance(h, str) or h is None:
        h, w = default_h, default_w
    return inputs[0].name, h, w, inputs[0].type


def get_output_name(session):
    return session.get_outputs()[0].name


def run_with_iobinding(session, input_name, output_name, input_array):
    """
    Execute inference using IOBinding, which pins the input tensor
    directly on the GPU and avoids a CPU→GPU copy at inference time.
    This is important for fair comparison with the two-stage script,
    which uses the same mechanism.
    """
    io_binding = session.io_binding()
    input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_array, 'cuda', 0)
    io_binding.bind_ortvalue_input(input_name, input_ortvalue)
    io_binding.bind_output(output_name, 'cuda', 0)
    session.run_with_iobinding(io_binding)
    return [io_binding.get_outputs()[0].numpy()]


# ── Power reading helper ───────────────────────────────────────────────────────

def read_vdd_in_mw(jetson):
    """
    Read total board input power (VDD_IN) in milliwatts from a live
    jtop instance. The key layout changed slightly between JetPack
    versions, so we try both known locations and fall back gracefully.
    """
    try:
        # JetPack 6.x layout
        return jetson.power['tot']['power']
    except (KeyError, TypeError):
        pass
    try:
        # Older JetPack layout
        return jetson.stats['VDD_IN']
    except (KeyError, TypeError):
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ONNX Baseline Inference with Data Collection')
    parser.add_argument('--source',   default='footage1_aigen.mp4',
                        help='Path to the video source file')
    parser.add_argument('--headless', action='store_true',
                        help='Disable display; recommended for benchmarking '
                             'so rendering cost does not affect measurements')
    parser.add_argument('--output',   default='results_onnx_baseline.json',
                        help='Path to save the collected measurement data')
    args = parser.parse_args()

    ort.set_default_logger_severity(3)   # suppress verbose TensorRT logs

    model_path = "optimised_yolo11/final_training_run/weights/best.onnx"

    print(f"[+] Loading session: {model_path}")
    sess = load_session(model_path,
                        trt_ep_context_file_path='./optimised_yolo11/final_training_run/weights')
    in_name, in_h, in_w, in_type = get_input_info(sess, default_h=1280, default_w=1280)
    out_name = get_output_name(sess)

    print(f"[+] Opening source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[!] Could not open source: {args.source}")
        return

    # ── Dummy warm-up ──────────────────────────────────────────────────────────
    # These passes handle CUDA context initialisation and initial weight
    # loading.  They are separate from the WARMUP_FRAMES exclusion below.
    print("[+] Running dummy warm-up passes...")
    dtype = np.float16 if '16' in str(in_type) else np.float32
    dummy = np.zeros((1, 3, in_h, in_w), dtype=dtype)
    for _ in range(5):
        run_with_iobinding(sess, in_name, out_name, dummy)
    print("[+] Dummy warm-up complete.")

    # ── Measurement storage ────────────────────────────────────────────────────
    latencies_ms   = []   # wall-clock latency per measured frame (ms)
    power_mw       = []   # VDD_IN power per measured frame (mW)
    frame_count    = 0    # total frames read (including warm-up exclusion)
    measured_count = 0    # frames that contribute to statistics

    target_w = 1280   # display width (only used if not headless)

    # ── jtop context ───────────────────────────────────────────────────────────
    # jtop runs a background monitoring thread; wrapping the entire inference
    # loop inside 'with jtop() as jetson' keeps it active throughout.
    # If jtop is unavailable the loop still runs, just without power data.
    def run_loop(jetson=None):
        nonlocal frame_count, measured_count

        start_total = time.time()

        while cap.isOpened():
            # ── Frame timing starts here ────────────────────────────────────
            # We time from cap.read() so that frame decoding cost is included,
            # consistent with the two-stage script's measurement window.
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess → inference → postprocess
            img, ratio, pad = preprocess(frame, in_h, in_w, in_type)
            outputs = run_with_iobinding(sess, in_name, out_name, img)
            boxes, scores, class_ids = postprocess(outputs, ratio, pad,
                                                   conf_thres=0.3, iou_thres=0.45)

            # ── Frame timing ends here ──────────────────────────────────────
            t_end  = time.time()
            frame_ms = (t_end - t_start) * 1000.0

            frame_count += 1

            # Skip the first WARMUP_FRAMES frames from the recorded statistics.
            # TensorRT continues kernel autotuning on early real inputs, so
            # their latency is not representative of steady-state performance.
            if frame_count > WARMUP_FRAMES:
                latencies_ms.append(frame_ms)

                # Read power immediately after timing so the sample is
                # temporally close to the inference window.
                if jetson is not None:
                    pwr = read_vdd_in_mw(jetson)
                    if pwr is not None:
                        power_mw.append(pwr)

                measured_count += 1

            # ── Optional display ────────────────────────────────────────────
            if not args.headless:
                fps_display = 1000.0 / frame_ms if frame_ms > 0 else 0
                for i in range(len(boxes)):
                    xmin, ymin, xmax, ymax = map(int, boxes[i])
                    xmin = max(0, xmin); ymin = max(0, ymin)
                    xmax = min(frame.shape[1], xmax); ymax = min(frame.shape[0], ymax)
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    label = f"Cls {class_ids[i]}: {scores[i]:.2f}"
                    cv2.putText(frame, label, (xmin, max(10, ymin - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, f"FPS: {fps_display:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                h, w = frame.shape[:2]
                if w > 0:
                    scale = target_w / float(w)
                    display = cv2.resize(frame, (target_w, int(h * scale)),
                                        interpolation=cv2.INTER_AREA)
                else:
                    display = frame

                cv2.imshow("ONNX Baseline", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        total_elapsed = time.time() - start_total
        return total_elapsed

    # Run the loop with or without jtop depending on availability
    if JTOP_AVAILABLE:
        with jtop() as jetson:
            total_elapsed = run_loop(jetson)
    else:
        total_elapsed = run_loop(jetson=None)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

    # ── Summary statistics ─────────────────────────────────────────────────────
    # All statistics are derived from the per-frame lists, not from the
    # overall elapsed time, because the overall time includes the warm-up
    # exclusion window and any teardown overhead.
    print(f"\n{'='*55}")
    print(f"  ONNX Baseline — Measurement Summary")
    print(f"{'='*55}")
    print(f"  Total frames read :  {frame_count}")
    print(f"  Warm-up excluded  :  {WARMUP_FRAMES}")
    print(f"  Measured frames   :  {measured_count}")

    results = {
        "configuration": "ONNX Baseline (FP16, TensorRT EP)",
        "model": model_path,
        "source": args.source,
        "warmup_frames_excluded": WARMUP_FRAMES,
        "total_frames_read": frame_count,
        "measured_frames": measured_count,
    }

    if latencies_ms:
        lat = np.array(latencies_ms)
        mean_fps   = 1000.0 / np.mean(lat)
        median_lat = float(np.median(lat))
        p95_lat    = float(np.percentile(lat, 95))
        std_fps    = float(np.std(1000.0 / lat))

        print(f"\n  Throughput & Latency")
        print(f"    Mean FPS         : {mean_fps:.2f}")
        print(f"    FPS std dev      : {std_fps:.2f}")
        print(f"    Median latency   : {median_lat:.2f} ms")
        print(f"    95th pct latency : {p95_lat:.2f} ms")

        results.update({
            "mean_fps": round(mean_fps, 3),
            "fps_std": round(std_fps, 3),
            "median_latency_ms": round(median_lat, 3),
            "p95_latency_ms": round(p95_lat, 3),
            "per_frame_latencies_ms": [round(v, 3) for v in latencies_ms],
        })

    if power_mw:
        pwr = np.array(power_mw)
        mean_pwr = float(np.mean(pwr))
        # Energy per frame: if mean power is in mW and mean FPS is frames/s,
        # then mW / FPS = mJ/frame  (since mW = mJ/s and 1/(frames/s) = s/frame)
        energy_mj = mean_pwr / mean_fps if latencies_ms else None

        print(f"\n  Power & Energy")
        print(f"    Mean VDD_IN power : {mean_pwr:.0f} mW")
        if energy_mj is not None:
            print(f"    Energy per frame  : {energy_mj:.2f} mJ")

        results.update({
            "mean_power_mw": round(mean_pwr, 1),
            "energy_per_frame_mj": round(energy_mj, 3) if energy_mj else None,
            "per_frame_power_mw": [round(v, 1) for v in power_mw],
        })
    else:
        print("\n  [!] No power data recorded (jtop unavailable or key mismatch).")

    print(f"{'='*55}\n")

    # Save to JSON so results can be loaded later for table/figure generation
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[+] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
