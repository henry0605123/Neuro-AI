"""對一組 LOO fold 結果的 DER(以及 FA/MD/SC)做 bootstrap 重抽信賴區間。

用法:
  python scripts/bootstrap_ci.py <結果目錄> [--exclude <session關鍵字>,... ] [--n 10000] [--seed 42]

會讀 <結果目錄>/fold*.json(每折一個 {"session":..., "DER":..., "FA":..., "MD":..., "SC":...}),
對折做 bootstrap 重抽 N 次(每次重抽 with replacement、算平均),回報:
  - 原始樣本平均(point estimate)
  - bootstrap 分布的 25% / 50%(中位數) / 75% 分位數
供之後每個 diarization 任務出結果時統一套用。
"""
import argparse, glob, json, os
import numpy as np


def bootstrap_ci(values, n=10000, seed=42, qs=(25, 50, 75)):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boot_means = values[idx].mean(axis=1)
    pct = np.percentile(boot_means, qs)
    return float(values.mean()), {f"{q}%": float(p) for q, p in zip(qs, pct)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--exclude", default="", help="逗號分隔,排除含這些字串的 session")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    exclude = tuple(x for x in args.exclude.split(",") if x)
    files = sorted(glob.glob(os.path.join(args.results_dir, "fold*.json")))
    rows = []
    excluded_rows = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if any(x in d.get("session", "") for x in exclude):
            excluded_rows.append(d)
        else:
            rows.append(d)

    if not rows:
        print("沒有可用的 fold 結果。")
        return

    print(f"共 {len(rows)} 折納入計算" + (f"(另有 {len(excluded_rows)} 折被排除: "
          + ", ".join(r['session'][9:][:8] for r in excluded_rows) + ")" if excluded_rows else ""))
    print(f"Bootstrap: n={args.n}, seed={args.seed}\n")

    print(f"{'指標':<6}{'平均值':>10}{'25%':>10}{'50%(中位)':>12}{'75%':>10}")
    for metric in ["DER", "FA", "MD", "SC"]:
        vals = [r[metric] for r in rows]
        mean, pct = bootstrap_ci(vals, n=args.n, seed=args.seed)
        print(f"{metric:<6}{mean:>10.2f}{pct['25%']:>10.2f}{pct['50%']:>12.2f}{pct['75%']:>10.2f}")

    if excluded_rows:
        print("\n被排除的折(單獨列出,不列入平均):")
        for r in excluded_rows:
            print(f"  {r['session'][9:][:10]:<12} DER={r['DER']:.1f} FA={r['FA']:.1f} "
                  f"MD={r['MD']:.1f} SC={r['SC']:.1f}")


if __name__ == "__main__":
    main()
