"""Compare accuracy + effective-rank evolution ACROSS runs, with quality gating.

Finds every history.json under the given run dirs / globs, keeps only the arms that
clear a quality bar (so collapsed arms don't clutter the plot), and plots probe accuracy
and effective rank vs epoch on a LOG-x axis -> PNG.

Gating (an arm is kept only if BOTH hold at some epoch):
    best test_acc >= --min-acc   (default 30 %)
    best erank    >= --min-erank (default 5)

Usage (run on the node, where runs/ lives):
    python plot_compare.py runs                      # all arms under runs/, gated
    python plot_compare.py runs/w512v2 runs/z2diag runs/randconv_feat2z
    python plot_compare.py "runs/**" --min-acc 30 --min-erank 5 --smooth 10 --out cmp.png

Headless-safe (Agg backend).
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


def find_histories(paths):
    """Expand each path: a history.json, a dir (recurse), or a glob -> sorted unique list."""
    found = []
    for p in paths:
        if os.path.isfile(p) and p.endswith("history.json"):
            found.append(p)
        elif os.path.isdir(p):
            found += glob.glob(os.path.join(p, "**", "history.json"), recursive=True)
        else:  # treat as a glob (may expand to dirs or files)
            for g in glob.glob(p, recursive=True):
                if os.path.isfile(g) and g.endswith("history.json"):
                    found.append(g)
                elif os.path.isdir(g):
                    found += glob.glob(os.path.join(g, "**", "history.json"), recursive=True)
    return sorted(set(found))


def label_for(f, root_parts):
    """Readable arm label: the run-dir path minus the common prefix, slashes -> '/'."""
    d = os.path.dirname(f)
    parts = os.path.normpath(d).split(os.sep)
    # drop the shared leading parts so labels stay short but unambiguous
    i = 0
    while i < len(root_parts) and i < len(parts) - 1 and parts[i] == root_parts[i]:
        i += 1
    return "/".join(parts[i:]) or os.path.basename(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="run dirs, history.json files, or globs")
    p.add_argument("--min-acc", type=float, default=30.0, help="keep arms reaching this test acc %% (default 30)")
    p.add_argument("--min-erank", type=float, default=5.0, help="keep arms reaching this erank (default 5)")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--out", default="compare.png")
    a = p.parse_args()

    files = find_histories(a.paths)
    if not files:
        print(f"no history.json found under: {a.paths}"); return
    root_parts = os.path.normpath(os.path.commonpath([os.path.dirname(f) for f in files])).split(os.sep)

    kept, dropped = [], []
    for f in files:
        try:
            h = json.load(open(f))
        except Exception as e:
            print(f"skip {f}: {e}"); continue
        if not h:
            continue
        best_acc = max(e["test_acc"] * 100 for e in h)
        best_er = max(e.get("erank", 0) for e in h)
        name = label_for(f, root_parts)
        if best_acc >= a.min_acc and best_er >= a.min_erank:
            kept.append((name, h, best_acc, best_er))
        else:
            dropped.append((name, best_acc, best_er))

    if not kept:
        print(f"no arm cleared acc>={a.min_acc} AND erank>={a.min_erank}. "
              f"Best seen: " + ", ".join(f"{n}(acc={ac:.0f},er={er:.1f})" for n, ac, er in dropped))
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, h, _, _ in kept:
        ep = [e["epoch"] for e in h]
        ax1.plot(ep, smooth([e["test_acc"] * 100 for e in h], a.smooth), label=name)
        ax2.plot(ep, smooth([e.get("erank", 0) for e in h], a.smooth), label=name)

    for ax in (ax1, ax2):
        ax.set_xscale("log"); ax.set_xlabel("epoch (log)"); ax.grid(alpha=.3, which="both")
    ax1.axhline(a.min_acc, ls="--", c="gray", lw=.8)
    ax2.axhline(a.min_erank, ls="--", c="gray", lw=.8)
    ax1.set_title(f"probe accuracy (smooth={a.smooth})  [gated acc>={a.min_acc:g}, erank>={a.min_erank:g}]")
    ax1.set_ylabel("acc %")
    ax2.set_title("effective rank of z"); ax2.set_ylabel("erank")
    ax1.legend(fontsize=7, ncol=2); ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}  ({len(kept)} arms kept, {len(dropped)} gated out)")
    if dropped:
        print("  gated out: " + ", ".join(f"{n}(acc={ac:.0f},er={er:.1f})" for n, ac, er in dropped))


if __name__ == "__main__":
    main()
