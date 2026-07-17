"""把 summary.json 裡的 bootstrap 信賴區間(p25/p50/p75)畫成分組長條圖 + 誤差線
(標準簡報風格:長條高度=中位數,誤差線=25%~75% 重抽區間)。

X 軸 = 三個指標(整體/成人/兒童),每組底下並排放各實驗的長條(顏色區分實驗)。

用法:
  python visualize_ci.py <summary1.json> [<label1>] <summary2.json> [<label2>] ...
  沒給 label 就用檔名。

輸出: outputs/ci_comparison.png
"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Microsoft JhengHei"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

METRICS = [("overall", "整體"), ("adult", "成人"), ("child", "兒童")]
PALETTE = ["#4472C4", "#C0504D", "#9BBB59", "#8064A2", "#4BACC6", "#F79646"]


def load_experiments(argv):
    """支援 'path.json' 或 'path.json label' 交錯輸入。"""
    exps = []
    i = 0
    while i < len(argv):
        path = argv[i]
        label = None
        if i + 1 < len(argv) and not argv[i + 1].endswith(".json"):
            label = argv[i + 1]
            i += 2
        else:
            i += 1
        d = json.load(open(path, encoding="utf-8"))
        name = label or os.path.splitext(os.path.basename(path))[0]
        exps.append((name, d))
    return exps


def main():
    argv = sys.argv[1:]
    if not argv:
        print("用法: python visualize_ci.py <summary.json> [label] ...")
        sys.exit(1)
    exps = load_experiments(argv)
    n_exp = len(exps)
    n_group = len(METRICS)

    fig, ax = plt.subplots(figsize=(3.2 * n_group + 1.5, 6))

    bar_w = 0.8 / n_exp
    group_centers = np.arange(n_group)

    for ei, (exp_name, d) in enumerate(exps):
        ci = d.get("bootstrap_ci_pct25_50_75", {})
        heights, err_lo, err_hi = [], [], []
        for key, _ in METRICS:
            m = ci.get(key)
            if not m:
                heights.append(0); err_lo.append(0); err_hi.append(0); continue
            p25, p50, p75 = m["p25"] * 100, m["p50"] * 100, m["p75"] * 100
            heights.append(p50)
            err_lo.append(p50 - p25)
            err_hi.append(p75 - p50)
        x = group_centers - 0.4 + bar_w * (ei + 0.5)
        color = PALETTE[ei % len(PALETTE)]
        ax.bar(x, heights, width=bar_w * 0.92, color=color, label=exp_name,
              edgecolor="white", linewidth=0.5, zorder=2)
        ax.errorbar(x, heights, yerr=[err_lo, err_hi], fmt="none",
                   ecolor="#404040", elinewidth=1.3, capsize=5, capthick=1.3, zorder=3)
        for xi, h, eh in zip(x, heights, err_hi):
            ax.text(xi, h + eh + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(group_centers)
    ax.set_xticklabels([zh for _, zh in METRICS], fontsize=12)
    ax.set_ylabel("CER (%)", fontsize=11)
    ax.set_title("各實驗 CER 比較(長條=重抽中位數,誤差線=25%–75% 重抽區間)",
                fontsize=12, pad=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, max(1, ax.get_ylim()[1]) * 1.12)

    plt.tight_layout()
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ci_comparison.png")
    plt.savefig(out_path, dpi=160)
    print(f"已存檔 -> {out_path}")


if __name__ == "__main__":
    main()
