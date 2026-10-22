"""Merge many run dirs into ONE table (+ optional plot) of final/best acc and late erank.

Usage:
  python summarize.py runs/w512v2 runs/z2diag                 # combined table
  python summarize.py runs/w512v2 runs/z2diag --plot out.png  # + acc/erank evolution plot
  python summarize.py runs/* --n 20                           # trailing window = 20

Reads each arm's result.json (if finished) and/or history.json (for in-progress + erank).
Rows are prefixed by their run dir so arms from different sweeps don't collide.
"""
import argparse, glob, json, os


def load(rundir, N):
    rows = []
    for hf in sorted(glob.glob(os.path.join(rundir, "*", "history.json"))):
        h = json.load(open(hf))
        if not h:
            continue
        d = os.path.dirname(hf)
        name = os.path.basename(d)
        rj = os.path.join(d, "result.json")
        done = os.path.exists(rj)
        r = json.load(open(rj)) if done else {}
        w = h[-N:]
        rows.append(dict(
            group=os.path.basename(rundir.rstrip("/")),
            arm=name, ep=h[-1]["epoch"], done=done,
            final=(r.get("final_acc") or h[-1]["test_acc"]) * 100,
            best=(r.get("best_acc") or max(e["test_acc"] for e in h)) * 100,
            acc_mean=sum(e["test_acc"] for e in w) / len(w) * 100,
            erank=sum(e.get("erank", 0) for e in w) / len(w),
            hist=h,
        ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--n", type=int, default=10, help="trailing window for means")
    p.add_argument("--plot", default=None, help="write an evolution PNG here")
    a = p.parse_args()

    rows = []
    for d in a.dirs:
        rows += load(d, a.n)
    rows.sort(key=lambda r: r["best"], reverse=True)

    print(f"{'group':12}{'arm':24}{'ep':>5}{'final':>8}{'best':>8}{'acc_m'+str(a.n):>8}{'erank':>7}  done")
    for r in rows:
        print(f"{r['group']:12}{r['arm']:24}{r['ep']:5d}{r['final']:8.2f}{r['best']:8.2f}"
              f"{r['acc_mean']:8.2f}{r['erank']:7.1f}  {'y' if r['done'] else '.'}")

    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        for r in rows:
            ep = [e["epoch"] for e in r["hist"]]
            ax1.plot(ep, [e["test_acc"] * 100 for e in r["hist"]], label=f"{r['group']}/{r['arm']}")
            ax2.plot(ep, [e.get("erank", 0) for e in r["hist"]], label=f"{r['group']}/{r['arm']}")
        ax1.set_title("probe acc"); ax1.set_xlabel("epoch"); ax1.set_ylabel("%"); ax1.grid(alpha=.3)
        ax2.set_title("erank(z)"); ax2.set_xlabel("epoch"); ax2.grid(alpha=.3)
        ax1.legend(fontsize=6, ncol=2); ax2.legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)
        print(f"wrote {a.plot}")


if __name__ == "__main__":
    main()
