import json
from pathlib import Path
import matplotlib.pyplot as plt


def main(json_path: str, out_dir: str, out_name: str = "log.png"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    with open(json_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    logs = sorted(logs, key=lambda x: int(x["epoch"]))

    epochs = [x["epoch"] for x in logs]
    losses = [x["avg_loss"] for x in logs]
    lrs = [x["lr"] for x in logs]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # ---- Avg Loss (LEFT AXIS) ----
    loss_color = "tab:blue"
    line_loss, = ax1.plot(
        epochs,
        losses,
        color=loss_color,
        marker="o",
        markersize=3,
        linewidth=1,
        label="Avg Loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Loss", color=loss_color)
    ax1.tick_params(axis="y", labelcolor=loss_color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- Learning Rate (RIGHT AXIS, LOG SCALE) ----
    ax2 = ax1.twinx()
    lr_color = "tab:red"
    line_lr, = ax2.plot(
        epochs,
        lrs,
        color=lr_color,
        linestyle="--",
        linewidth=2,
        label="Learning Rate (log)",
    )
    ax2.set_ylabel("Learning Rate", color=lr_color)
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor=lr_color)

    # ---- Title ----
    fig.suptitle("Training Trends: Avg Loss & Learning Rate")

    # ---- Legend (RIGHT SIDE) ----
    fig.legend(
        handles=[line_loss, line_lr],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    fig.savefig(out_path, dpi=200)
    print(f"[OK] Saved plot to {out_path}")


if __name__ == "__main__":
    main(
        json_path="/home/soic/Desktop/tracking/v24/training/tracking/res/checkpoints/log/training_log_20260106_163601.json",
        out_dir="/home/soic/Desktop/tracking/v24/training/tracking/res/checkpoints/test_6_with_mosic_1000_fine",
    )
