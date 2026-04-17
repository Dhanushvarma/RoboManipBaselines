import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt

SRC_ROOT = "../dataset/MujocoCallMTrash_20260417_130627"
DST_ROOT = "../dataset/MujocoCallMTrash_20260417_130627_under_30"

THRESHOLD_SEC = 30.0

os.makedirs(DST_ROOT, exist_ok=True)

durations = []
demo_durations = {}


kept = 0
skipped = 0

# --- Scan dataset once ---
for demo in os.listdir(SRC_ROOT):
    src_demo_path = os.path.join(SRC_ROOT, demo)

    if not os.path.isdir(src_demo_path):
        continue

    video_path = os.path.join(src_demo_path, "base_rgb_image.rmb.mp4")

    if not os.path.exists(video_path):
        print(f"[WARN] Missing video: {demo}")
        skipped += 1
        continue

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:
        print(f"[WARN] Bad FPS: {demo}")
        skipped += 1
        continue

    duration = frame_count / fps

    durations.append(duration)
    demo_durations[demo] = duration

# --- Convert to numpy ---
durations = np.array(durations)

# --- Stats ---
print("\n=== Dataset Stats ===")
print("Num demos:", len(durations))
print("Min:", durations.min())
print("Max:", durations.max())
print("Mean:", durations.mean())
print("Median:", np.median(durations))

# --- Buckets ---
bins = np.arange(0, 70, 10)
hist, edges = np.histogram(durations, bins=bins)

print("\nDemo duration buckets:")
for i in range(len(hist)):
    start = int(edges[i])
    end = int(edges[i + 1])
    print(f"{start:2d}–{end:2d} sec: {hist[i]} demos")

# --- Copy under threshold ---
for demo, duration in demo_durations.items():
    if duration < THRESHOLD_SEC:
        src_demo_path = os.path.join(SRC_ROOT, demo)
        dst_demo_path = os.path.join(DST_ROOT, demo)

        shutil.copytree(src_demo_path, dst_demo_path, dirs_exist_ok=True)

        print(f"[COPY] {demo} ({duration:.2f}s)")
        kept += 1
    else:
        skipped += 1

print("\n=== Copy Summary ===")
print(f"Copied (<{THRESHOLD_SEC}s): {kept}")
print(f"Skipped: {skipped}")

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.hist(durations, bins=20, edgecolor="black")

plt.axvline(durations.mean(), linestyle="--", label=f"Mean: {durations.mean():.2f}s")
plt.axvline(durations.min(), linestyle=":", label=f"Min: {durations.min():.2f}s")
plt.axvline(durations.max(), linestyle=":", label=f"Max: {durations.max():.2f}s")

plt.xlabel("Demo Duration (seconds)")
plt.ylabel("Count")
plt.title("Demo Length Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("demo_duration_hist.png", dpi=150)

print("\nSaved histogram to demo_duration_hist.png")
