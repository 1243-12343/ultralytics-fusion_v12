import numpy as np
import matplotlib.pyplot as plt

def minmax_norm(arr, higher_is_better=True, eps=1e-12):
    """Min-Max normalize to [0, 1]. If higher_is_better=False, invert after normalization."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    norm = (arr - mn) / (mx - mn + eps)
    return norm if higher_is_better else (1.0 - norm)

def radar_chart(models, metrics, values, title="Radar Chart", save_path="radar_hexagram.png"):
    """
    models: list[str]
    metrics: list[str] (len = N axes)
    values: np.ndarray shape (num_models, N axes), already normalized to [0,1]
    """
    num_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]  # close

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=90, fontsize=9)

    # Style
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # Plot each model
    for i, m in enumerate(models):
        data = values[i].tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=2, label=m)
        ax.fill(angles, data, alpha=0.08)

    ax.set_title(title, fontsize=14, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    # -----------------------------
    # Raw table data (your numbers)
    # -----------------------------
    models = ["YOLOV8", "YOLOV11", "YOLOV13", "YOLOV12", "Ours"]

    map50 = np.array([73.22, 73.41, 73.53, 74.29, 77.05])
    map75 = np.array([35.72, 35.98, 35.86, 36.82, 38.45])
    map5095 = np.array([37.90, 38.73, 38.74, 39.17, 40.36])
    param_m = np.array([2.69, 2.59, 2.46, 2.56, 2.93])
    gflops = np.array([6.9, 6.4, 6.3, 6.0, 7.3])

    # 6th axis to form "六芒星"
    acc_avg = (map50 + map75 + map5095) / 3.0

    # Axes (6 points)
    metrics = ["mAP50", "mAP75", "mAP50-95", "AccAvg", "Param(↓)", "GFLOPs(↓)"]

    # Normalize: higher better for accuracy metrics, lower better for complexity metrics
    n_map50 = minmax_norm(map50, higher_is_better=True)
    n_map75 = minmax_norm(map75, higher_is_better=True)
    n_map5095 = minmax_norm(map5095, higher_is_better=True)
    n_accavg = minmax_norm(acc_avg, higher_is_better=True)
    n_param = minmax_norm(param_m, higher_is_better=False)  # lower is better
    n_gflops = minmax_norm(gflops, higher_is_better=False)  # lower is better

    values = np.stack([n_map50, n_map75, n_map5095, n_accavg, n_param, n_gflops], axis=1)

    radar_chart(
        models=models,
        metrics=metrics,
        values=values,
        title="Comparison of Models (Normalized) - Hexagram Radar",
        save_path="model_hexagram_radar.png",
    )