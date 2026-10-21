"""Plot accuracy + effective-rank evolution for all arms in a runs/ dir -> PNG.

Usage:  python plot_evolution.py runs/w512v2            # -> runs/w512v2/evolution.png
        python plot_evolution.py runs/w512v2 --smooth 10 --out /tmp/evo.png
Headless-safe (Agg backend). Open the PNG however you view images on the node.
"""
import argparse, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(ys, k):
    if k <= 1:
        return ys
    out = []
    for i in range(len(ys)):
        lo = max(0, i - k + 1)
        out.append(sum(ys[lo:i + 1]) / (i + 1 - lo))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundir")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    runs = sorted(glob.glob(os.path.join(a.rundir, "*", "history.json")))
    if not runs:
        print(f"no history.json under {a.rundir}"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for f in runs:
        h = json.load(open(f))
        if not h:
            continue
        name = os.path.basename(os.path.dirname(f))
        ep = [e["epoch"] for e in h]
        acc = smooth([e["test_acc"] * 100 for e in h], a.smooth)
        er = smooth([e.get("erank", 0) for e in h], a.smooth)
        ax1.plot(ep, acc, label=name)
        ax2.plot(ep, er, label=name)

    ax1.set_title(f"probe accuracy (smooth={a.smooth})"); ax1.set_xlabel("epoch"); ax1.set_ylabel("acc %")
    ax2.set_title("effective rank of z"); ax2.set_xlabel("epoch"); ax2.set_ylabel("erank")
    ax1.grid(alpha=.3); ax2.grid(alpha=.3)
    ax1.legend(fontsize=7, ncol=2); ax2.legend(fontsize=7, ncol=2)
    out = a.out or os.path.join(a.rundir, "evolution.png")
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
