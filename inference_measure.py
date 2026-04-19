"""
inference_measure.py  —  PyTorch Baseline Inference with Data Collection
================================================================
Runs the single-model YOLO11n PyTorch baseline on a video source and
collects per-frame wall-clock latency and power data for thesis evaluation.

KEY DESIGN DECISION — why we read frames ourselves instead of passing
the video path to model.predict():
  When you call model.predict(video_path, stream=True), Ultralytics
  manages the frame-reading loop internally and you cannot place a
  timer around the full per-frame pipeline.  The r.speed dictionary
  it exposes only covers its own preprocess/inference/postprocess time
  and excludes Python overhead between frames, making it incomparable
  to the ONNX scripts' wall-clock measurements.

  By reading frames with cv2.VideoCapture ourselves and calling
  model.predict(frame) on each individual frame, we get the same
  timing window as the ONNX scripts: start timer → cap.read() →
  model.predict() → end timer.  This produces numbers that can be
  placed in the same thesis table as the ONNX baseline results.

Measurement methodology (matches inference_onnx.py):
  - Wall-clock latency: time from cap.read() to end of model.predict(),
    covering the full per-frame cost visible to a downstream consumer.
  - Power (VDD_IN): read from jtop at the end of each measured frame.
  - First WARMUP_FRAMES frames are excluded from statistics to allow
    the model's internal CUDA kernel autotuning to stabilise.
  - Results are saved to a JSON file for later analysis.
"""

import cv2
import numpy as np
import time
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    print("[!] jtop not found. Power measurements will not be recorded.")
    print("    Install with: pip install jetson-stats")
    JTOP_AVAILABLE = False

# Number of real-input frames to discard at the start of the run.
# Even after the model is loaded, CUDA JIT compilation and memory
# allocation for the first real input shapes can inflate early latency.
WARMUP_FRAMES = 30


# ── Power reading helper ───────────────────────────────────────────────────────

def read_vdd_in_mw(jetson):
    """
    Read total board input power (VDD_IN) in milliwatts from a live
    jtop instance.  Tries the JetPack 6.x key layout first, then the
    older layout as a fallback.  Returns None if neither key is found,
    which causes the caller to simply skip recording power for that frame
    rather than crashing.
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
    parser = argparse.ArgumentParser(
        description='PyTorch Baseline Inference with Wall-Clock Data Collection')
    parser.add_argument('--source',
                        default='footage1_aigen.mp4',
                        help='Path to the video source file')
    parser.add_argument('--headless',
                        action='store_true',
                        help='Disable display; strongly recommended for '
                             'benchmarking so rendering overhead is excluded '
                             'from the timing window')
    parser.add_argument('--output',
                        default='results_pytorch_baseline.json',
                        help='Path to save the collected measurement data')
    args = parser.parse_args()

    model_path = "optimised_yolo11/final_training_run/weights/best.pt"

    # Load the model.  The first call to model.predict() after loading
    # will trigger CUDA context initialisation and JIT compilation;
    # the WARMUP_FRAMES exclusion below absorbs the cost of these one-off
    # operations so they do not inflate the reported latency distribution.
    print(f"[+] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[+] Opening source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[!] Could not open source: {args.source}")
        return

    # ── Measurement storage ────────────────────────────────────────────────────
    latencies_ms   = []   # wall-clock per-frame latency in ms
    power_mw       = []   # VDD_IN power per measured frame in mW
    frame_count    = 0    # total frames processed (including warm-up exclusion)
    measured_count = 0    # frames contributing to statistics

    target_w = 1280       # display width (only used when not headless)

    def run_loop(jetson=None):
        nonlocal frame_count, measured_count

        start_total = time.time()

        while cap.isOpened():
            # ── Frame timing starts here ────────────────────────────────────
            # We include cap.read() in the timing window for consistency with
            # the ONNX scripts, which also time from cap.read() onwards.
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the single frame.
            # verbose=False suppresses Ultralytics' per-frame console output,
            # which would otherwise add a small but real I/O cost to every
            # frame and interfere with clean timing.
            # save=False, save_txt=False and save_crop=False ensure that no
            # disk writes occur inside the timing window, since disk I/O would
            # inflate latency and is absent in the ONNX scripts.
            results = model.predict(
                frame,
                verbose=False,
                save=False,
                save_txt=False,
                save_crop=False,
                conf=0.3,
                iou=0.45,
            )

            # Accessing results[0] triggers any lazy computation that
            # Ultralytics defers until the result is consumed.  We do this
            # inside the timing window so that deferred processing is counted.
            r = results[0]

            # ── Frame timing ends here ──────────────────────────────────────
            t_end    = time.time()
            frame_ms = (t_end - t_start) * 1000.0

            frame_count += 1

            # Exclude the first WARMUP_FRAMES from recorded statistics.
            if frame_count > WARMUP_FRAMES:
                latencies_ms.append(frame_ms)

                if jetson is not None:
                    pwr = read_vdd_in_mw(jetson)
                    if pwr is not None:
                        power_mw.append(pwr)

                measured_count += 1

            # ── Optional display ────────────────────────────────────────────
            if not args.headless:
                fps_display = 1000.0 / frame_ms if frame_ms > 0 else 0

                # r.plot() renders bounding boxes onto a copy of the frame.
                # We call it outside the timing window (the timer has already
                # stopped) so rendering cost does not affect measurements.
                annotated = r.plot(line_width=2)

                cv2.putText(annotated,
                            f"FPS: {fps_display:.1f}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                h, w = annotated.shape[:2]
                if w > 0:
                    scale   = target_w / float(w)
                    display = cv2.resize(annotated,
                                         (target_w, int(h * scale)),
                                         interpolation=cv2.INTER_AREA)
                else:
                    display = annotated

                cv2.imshow("PyTorch Baseline", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return time.time() - start_total

    # ── jtop context ───────────────────────────────────────────────────────────
    # Opening jtop before the loop keeps the background monitoring thread
    # alive throughout the entire run so that power samples are available
    # at every frame.  If jtop is unavailable we still collect latency data.
    if JTOP_AVAILABLE:
        with jtop() as jetson:
            total_elapsed = run_loop(jetson)
    else:
        total_elapsed = run_loop(jetson=None)

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

    # ── Summary statistics ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PyTorch Baseline — Measurement Summary")
    print(f"{'='*55}")
    print(f"  Total frames read :  {frame_count}")
    print(f"  Warm-up excluded  :  {WARMUP_FRAMES}")
    print(f"  Measured frames   :  {measured_count}")

    results_dict = {
        "configuration": "PyTorch Baseline (FP32, Ultralytics predict API)",
        "model": model_path,
        "source": args.source,
        "warmup_frames_excluded": WARMUP_FRAMES,
        "total_frames_read": frame_count,
        "measured_frames": measured_count,
        "note": (
            "Latency is wall-clock time from cap.read() to end of "
            "model.predict(), which includes Ultralytics result "
            "construction overhead in addition to preprocess/inference/"
            "postprocess.  This is comparable to the ONNX scripts' "
            "measurement window but slightly higher than Ultralytics' "
            "internal r.speed timing, which excludes result overhead."
        ),
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

        results_dict.update({
            "mean_fps": round(mean_fps, 3),
            "fps_std": round(std_fps, 3),
            "median_latency_ms": round(median_lat, 3),
            "p95_latency_ms": round(p95_lat, 3),
            "per_frame_latencies_ms": [round(v, 3) for v in latencies_ms],
        })

    if power_mw:
        pwr      = np.array(power_mw)
        mean_pwr   = float(np.mean(pwr))
        energy_mj  = mean_pwr / mean_fps if latencies_ms else None

        print(f"\n  Power & Energy")
        print(f"    Mean VDD_IN power : {mean_pwr:.0f} mW")
        if energy_mj is not None:
            print(f"    Energy per frame  : {energy_mj:.2f} mJ")

        results_dict.update({
            "mean_power_mw": round(mean_pwr, 1),
            "energy_per_frame_mj": round(energy_mj, 3) if energy_mj else None,
            "per_frame_power_mw": [round(v, 1) for v in power_mw],
        })
    else:
        print("\n  [!] No power data recorded (jtop unavailable or key mismatch).")

    print(f"{'='*55}\n")

    with open(args.output, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"[+] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
