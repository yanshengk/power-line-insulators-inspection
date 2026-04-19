"""
two_stage_inference_onnx_measure.py  —  Two-Stage ONNX Pipeline with Data Collection
=============================================================================
Runs the two-stage YOLO11n pipeline (Stage 1: string localisation,
Stage 2: disc-level defect detection) with binary Shannon entropy gating
on a video source, and collects per-frame latency, power, and Stage 2
routing statistics for thesis evaluation.

Measurement methodology:
  - Wall-clock latency: time from cap.read() to end of all Stage 2
    postprocessing, covering the full pipeline cost for every frame.
  - Power (VDD_IN): read from jtop at the end of each measured frame.
  - First WARMUP_FRAMES frames are excluded from all statistics because
    TensorRT kernel autotuning on real inputs produces anomalously high
    latency that is not representative of steady-state performance.
  - Stage 2 routing rate (fraction of detections forwarded to Stage 2)
    is also recorded, as this is a key result for the entropy gate
    analysis in Section 4.4 of the thesis.
  - Results are saved to a JSON file for later analysis.
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import math
import argparse
import json

try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    print("[!] jtop not found. Power measurements will not be recorded.")
    print("    Install with: pip install jetson-stats")
    JTOP_AVAILABLE = False

# Number of real-input frames to exclude from statistics at the start
# of the run, to allow TensorRT kernel autotuning to stabilise.
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
    img = np.expand_dims(img, axis=0)
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

    boxes     = predictions[:, :4]
    scores    = np.max(predictions[:, 4:], axis=1)
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    mask      = scores > conf_thres
    boxes     = boxes[mask]
    scores    = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], [], []

    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep      = nms(boxes_xyxy, scores, iou_thres)
    boxes_xyxy = boxes_xyxy[keep]
    scores     = scores[keep]
    class_ids  = class_ids[keep]

    boxes_xyxy[:, [0, 2]] -= pad[0]
    boxes_xyxy[:, [1, 3]] -= pad[1]
    boxes_xyxy[:, [0, 2]] /= ratio[0]
    boxes_xyxy[:, [1, 3]] /= ratio[1]

    return boxes_xyxy, scores, class_ids


# ── Uncertainty gate ───────────────────────────────────────────────────────────

def calculate_shannon_entropy(probability):
    """
    Compute binary Shannon entropy for a single probability value p.
    The result is in the range [0, 1] bits:
      - H = 0 when p = 0 or p = 1 (complete certainty)
      - H = 1 when p = 0.5 (maximum uncertainty)
    The clipping to [1e-7, 1 - 1e-7] avoids log(0) without meaningfully
    affecting the result for any realistic confidence score.
    """
    p = max(min(probability, 1.0 - 1e-7), 1e-7)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ── Session helpers ────────────────────────────────────────────────────────────

def load_session(model_path, trt_ep_context_file_path='./trt_engines'):
    """
    Load an ONNX model and configure ONNX Runtime to use the TensorRT
    Execution Provider with FP16 enabled and engine caching on disk.
    Thread counts are matched identically to the ONNX baseline script
    so that scheduling cannot explain any throughput difference.
    """
    sess_options = ort.SessionOptions()
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
    Execute inference using IOBinding for both models, consistent with
    the ONNX baseline.  Pinning inputs directly on the GPU avoids a
    CPU→GPU copy and is particularly valuable for Stage 2 crops, which
    are produced frequently and at varying sizes.
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
    Read total board input power (VDD_IN) in milliwatts from jtop.
    Tries the JetPack 6.x key first, then the older layout.
    """
    try:
        return jetson.power['tot']['power']
    except (KeyError, TypeError):
        pass
    try:
        return jetson.stats['VDD_IN']
    except (KeyError, TypeError):
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Two-Stage ONNX Inference with Data Collection')
    parser.add_argument('--source',    default='footage1_aigen.mp4',
                        help='Path to the video source file')
    parser.add_argument('--headless',  action='store_true',
                        help='Disable display; recommended for benchmarking')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Binary Shannon entropy threshold θ for the '
                             'uncertainty gate (default: 0.3)')
    parser.add_argument('--output',    default='results_two_stage.json',
                        help='Path to save the collected measurement data')
    args = parser.parse_args()

    ort.set_default_logger_severity(3)

    model1_path = "runs/detect/train16/weights/train16_best.onnx"
    model2_path = "runs/detect/train17/weights/train17_best.onnx"
    entropy_threshold = args.threshold

    print(f"[+] Loading Stage 1 session: {model1_path}")
    sess1 = load_session(model1_path,
                         trt_ep_context_file_path='./runs/detect/train16/weights')
    in1_name, in1_h, in1_w, in1_type = get_input_info(sess1)
    out1_name = get_output_name(sess1)

    print(f"[+] Loading Stage 2 session: {model2_path}")
    sess2 = load_session(model2_path,
                         trt_ep_context_file_path='./runs/detect/train17/weights')
    in2_name, in2_h, in2_w, in2_type = get_input_info(sess2)
    out2_name = get_output_name(sess2)

    names1 = {0: 'insulator string'}
    names2 = {0: 'flashed', 1: 'broken'}

    print(f"[+] Opening source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[!] Could not open source: {args.source}")
        return

    # ── Dummy warm-up ──────────────────────────────────────────────────────────
    print("[+] Running dummy warm-up passes for both engines...")
    dtype1 = np.float16 if '16' in str(in1_type) else np.float32
    dtype2 = np.float16 if '16' in str(in2_type) else np.float32
    dummy1 = np.zeros((1, 3, in1_h, in1_w), dtype=dtype1)
    dummy2 = np.zeros((1, 3, in2_h, in2_w), dtype=dtype2)
    for _ in range(5):
        run_with_iobinding(sess1, in1_name, out1_name, dummy1)
        run_with_iobinding(sess2, in2_name, out2_name, dummy2)
    print("[+] Dummy warm-up complete.")

    # ── Measurement storage ────────────────────────────────────────────────────
    latencies_ms   = []   # wall-clock latency per measured frame (ms)
    power_mw       = []   # VDD_IN power per measured frame (mW)

    # Routing counters for entropy gate analysis (Section 4.4 of thesis).
    # We track every detection across all measured frames so we can report
    # the overall Stage 2 routing rate as a percentage.
    total_detections   = 0   # total Stage 1 detections across all measured frames
    stage2_invocations = 0   # how many of those were forwarded to Stage 2

    frame_count    = 0
    measured_count = 0
    target_w       = 1280

    def run_loop(jetson=None):
        nonlocal frame_count, measured_count
        nonlocal total_detections, stage2_invocations

        start_total = time.time()

        while True:
            # ── Frame timing starts here ────────────────────────────────────
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # ── Stage 1: detect insulator strings ──────────────────────────
            img1, ratio1, pad1 = preprocess(frame, in1_h, in1_w, in1_type)
            outputs1 = run_with_iobinding(sess1, in1_name, out1_name, img1)
            boxes1, scores1, class1_ids = postprocess(
                outputs1, ratio1, pad1, conf_thres=0.3, iou_thres=0.45)

            # ── Entropy gate + Stage 2 crop preprocessing ──────────────────
            # We separate preprocessing from inference so that all crop
            # preprocessing is done before any Stage 2 inference call is
            # issued.  This mirrors the script's design and avoids mixing
            # preprocessing and inference costs in the timing.
            stage1_detections = []
            stage2_jobs       = []

            for i in range(len(boxes1)):
                x1, y1, x2, y2 = map(int, boxes1[i])
                conf1   = scores1[i]
                cls1_id = class1_ids[i]

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                entropy       = calculate_shannon_entropy(conf1)
                needs_stage2  = not (cls1_id == 0 and entropy < entropy_threshold)

                stage1_detections.append(
                    (x1, y1, x2, y2, conf1, cls1_id, entropy, needs_stage2))

                if needs_stage2:
                    crop = frame[y1:y2, x1:x2]
                    img2, ratio2, pad2 = preprocess(crop, in2_h, in2_w, in2_type)
                    stage2_jobs.append(
                        (len(stage1_detections) - 1, img2, ratio2, pad2))

            # ── Stage 2: detect defects in uncertain crops ──────────────────
            stage2_results = {}
            for det_idx, img2, ratio2, pad2 in stage2_jobs:
                outputs2 = run_with_iobinding(sess2, in2_name, out2_name, img2)
                boxes2, scores2, class2_ids = postprocess(
                    outputs2, ratio2, pad2, conf_thres=0.3, iou_thres=0.45)
                stage2_results[det_idx] = (boxes2, scores2, class2_ids)

            # ── Frame timing ends here ──────────────────────────────────────
            t_end    = time.time()
            frame_ms = (t_end - t_start) * 1000.0

            frame_count += 1

            if frame_count > WARMUP_FRAMES:
                latencies_ms.append(frame_ms)

                # Record power sample immediately after the timing window
                if jetson is not None:
                    pwr = read_vdd_in_mw(jetson)
                    if pwr is not None:
                        power_mw.append(pwr)

                # Accumulate routing statistics.
                # We record the number of valid Stage 1 detections and how
                # many were routed to Stage 2 so we can compute the routing
                # rate over the full measured run.
                n_valid    = len(stage1_detections)
                n_routed   = len(stage2_jobs)
                total_detections   += n_valid
                stage2_invocations += n_routed

                measured_count += 1

            # ── Optional display ────────────────────────────────────────────
            if not args.headless:
                fps_display = 1000.0 / frame_ms if frame_ms > 0 else 0
                for idx, (x1, y1, x2, y2, conf1, cls1_id, entropy, _) \
                        in enumerate(stage1_detections):
                    cls1_name = names1.get(cls1_id, str(cls1_id))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls1_name} {conf1:.2f} (H:{entropy:.2f})"
                    cv2.putText(frame, label, (x1, max(10, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if idx in stage2_results:
                        boxes2, scores2, class2_ids = stage2_results[idx]
                        for j in range(len(boxes2)):
                            cx1, cy1, cx2, cy2 = map(int, boxes2[j])
                            abs_x1 = x1 + cx1; abs_y1 = y1 + cy1
                            abs_x2 = x1 + cx2; abs_y2 = y1 + cy2
                            cv2.rectangle(frame,
                                          (abs_x1, abs_y1), (abs_x2, abs_y2),
                                          (0, 0, 255), 2)
                            cls2_name = names2.get(class2_ids[j], str(class2_ids[j]))
                            cv2.putText(frame,
                                        f"{cls2_name} {scores2[j]:.2f}",
                                        (abs_x1, max(10, abs_y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 2)

                cv2.putText(frame, f"FPS: {fps_display:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                h, w = frame.shape[:2]
                if w > 0:
                    scale   = target_w / float(w)
                    display = cv2.resize(frame, (target_w, int(h * scale)),
                                         interpolation=cv2.INTER_AREA)
                else:
                    display = frame

                cv2.imshow("ONNX Two-Stage", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return time.time() - start_total

    if JTOP_AVAILABLE:
        with jtop() as jetson:
            total_elapsed = run_loop(jetson)
    else:
        total_elapsed = run_loop(jetson=None)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

    # ── Summary statistics ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Two-Stage ONNX Pipeline — Measurement Summary")
    print(f"  Entropy threshold θ = {entropy_threshold}")
    print(f"{'='*60}")
    print(f"  Total frames read :  {frame_count}")
    print(f"  Warm-up excluded  :  {WARMUP_FRAMES}")
    print(f"  Measured frames   :  {measured_count}")

    results = {
        "configuration": "Two-Stage ONNX Pipeline (FP16, TensorRT EP)",
        "stage1_model": model1_path,
        "stage2_model": model2_path,
        "entropy_threshold": entropy_threshold,
        "source": args.source,
        "warmup_frames_excluded": WARMUP_FRAMES,
        "total_frames_read": frame_count,
        "measured_frames": measured_count,
    }

    if latencies_ms:
        lat      = np.array(latencies_ms)
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
        pwr      = np.array(power_mw)
        mean_pwr = float(np.mean(pwr))
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

    # ── Entropy gate routing statistics ───────────────────────────────────────
    # This is the key result for Section 4.4 of the thesis.  It tells you
    # what fraction of Stage 1 detections were forwarded to Stage 2 — a low
    # rate means most frames were processed by Stage 1 alone, keeping latency
    # close to the single-stage ONNX baseline for the majority of the run.
    print(f"\n  Entropy Gate Routing (θ = {entropy_threshold})")
    if total_detections > 0:
        routing_rate_pct = 100.0 * stage2_invocations / total_detections
        print(f"    Total Stage 1 detections  : {total_detections}")
        print(f"    Forwarded to Stage 2      : {stage2_invocations}")
        print(f"    Stage 2 routing rate      : {routing_rate_pct:.1f}%")
        print(f"    (Skipped by gate          : {100 - routing_rate_pct:.1f}%)")

        results.update({
            "total_stage1_detections": total_detections,
            "stage2_invocations": stage2_invocations,
            "stage2_routing_rate_pct": round(routing_rate_pct, 2),
        })
    else:
        print("    No Stage 1 detections recorded in measured frames.")

    print(f"{'='*60}\n")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[+] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
