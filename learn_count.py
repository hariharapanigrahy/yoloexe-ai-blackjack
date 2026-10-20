"""The AI discovers card counting by itself (nobody tells it any card values).
It plays hundreds of millions of rounds with its learned chart, and before every round notes which cards have already left the shoe.
Then it asks: when MORE of card X is gone, do I win more or less next round? (a linear fit over every round = 'effect of removal').
The answer is its own value for each card. Then it learns what its own count is worth, builds its bet ramp, and we compare with Hi-Lo.
  python learn_count.py [--run bj2] [--rounds 40000000] [--workers 10]   -> runs/<run>/count_weights.npy, count_ramp.npy, count_log.json"""
import os, sys, json, time, argparse, numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bj

PFULL = np.array([0, 0] + [4 / 52] * 8 + [16 / 52, 4 / 52])           # share of each value (2..11) in a full shoe

@njit(cache=True)
def gather(n, pol, seed, XtX, Xty, h17=False):
    """plays n rounds flat; before each round x[v] = (cards of value v already gone - expected) / decks left. Accumulates the fit."""
    np.random.seed(seed); shoe = bj.make_shoe(); np.random.shuffle(shoe); pos = 0; rc = 0
    seen = np.zeros(12); Q = np.zeros((1, 1, 1, 1)); x = np.zeros(11)
    for i in range(n):
        if pos > bj.CUT: np.random.shuffle(shoe); pos = 0; seen[:] = 0
        dl = (bj.NCARDS - pos) / 52.0
        for v in range(2, 12): x[v - 2] = (seen[v] - pos * PFULL[v]) / dl
        x[10] = 1.0
        r, p2, rc = bj.play_round(shoe, pos, rc, pol, 0.0, Q, Q, False, 1.0, h17)
        for k in range(pos, p2): seen[shoe[k]] += 1
        pos = p2
        for a in range(11):
            Xty[a] += x[a] * r
            for b in range(11): XtX[a, b] += x[a] * x[b]
    return 0

@njit(cache=True)
def play_count(n, pol, w, ramp, seed, ev, en):
    """plays n rounds counting with weights w[v] (true count = running / decks left), betting ramp[tc+10] (flat if ramp[0] < 0)."""
    np.random.seed(seed); shoe = bj.make_shoe(); np.random.shuffle(shoe); pos = 0; rc = 0; run = 0.0
    Q = np.zeros((1, 1, 1, 1)); tot = 0.0; wag = 0.0
    for i in range(n):
        if pos > bj.CUT: np.random.shuffle(shoe); pos = 0; run = 0.0
        tc = int(np.floor(run / max((bj.NCARDS - pos) / 52.0, 0.5))); tci = min(20, max(0, tc + 10))
        bet = 1.0 if ramp[0] < 0 else ramp[tci]
        r, p2, rc = bj.play_round(shoe, pos, rc, pol, 0.0, Q, Q, False, bet)
        for k in range(pos, p2): run += w[shoe[k]]
        pos = p2; tot += r; wag += bet; ev[tci] += r / bet; en[tci] += 1
    return tot, wag

def _gather(a):
    pol, seed, n = a[:3]; h17 = a[3] if len(a) > 3 else False; XtX = np.zeros((11, 11)); Xty = np.zeros(11); gather(n, pol, seed, XtX, Xty, h17); return XtX, Xty
def _play(a):
    pol, w, ramp, seed, n = a; ev = np.zeros(21); en = np.zeros(21); t, g = play_count(n, pol, w, ramp, seed, ev, en); return t, g, ev, en
def ramp_from(edge):                                                   # same rule as train.py
    return np.array([0.0001 if e < -0.01 else 1.0 if e <= 0 else float(min(12, max(1, round(e / 0.0022)))) for e in edge])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='bj2'); ap.add_argument('--rounds', type=int, default=40_000_000); ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args(); out = os.path.join(HERE, 'runs', a.run); pol = np.load(os.path.join(out, 'chart.npy')); t0 = time.time(); W = a.workers; per = a.rounds // W
    with ProcessPoolExecutor(W) as ex:
        R = list(ex.map(_gather, [(pol, 500 + k, per) for k in range(W)]))
        XtX = sum(r[0] for r in R); Xty = sum(r[1] for r in R)
        beta = np.linalg.solve(XtX + 1e-6 * np.eye(11), Xty)         # beta[v]: change in my result when one extra card v (per deck) is gone
        eor = beta[:10] - (beta[:10] * PFULL[2:]).sum()                # remove the shared part: only differences between cards matter
        w = np.zeros(12); w[2:] = eor / -eor[8]                        # its own scale: a ten = -1
        print('LEARNED CARD VALUES (+ = good for me when it is gone):', '  '.join(f"{'A' if v == 11 else v}:{w[v]:+.2f}" for v in range(2, 12)), f'  [{time.time() - t0:.0f}s]', flush=True)
        HILO = np.zeros(12); HILO[2:7] = 1; HILO[10:12] = -1; res = {}
        for name, ww in (('LEARNED', w), ('HI-LO (human)', HILO)):
            Rr = list(ex.map(_play, [(pol, ww, -np.ones(21), 900 + k, per // 2) for k in range(W)]))
            ev = sum(r[2] for r in Rr); en = sum(r[3] for r in Rr); edge = np.where(en > 0, ev / np.maximum(en, 1), 0.0); ramp = ramp_from(edge)
            Rr = list(ex.map(_play, [(pol, ww, ramp, 1900 + k, per // 2) for k in range(W)]))
            tot = sum(r[0] for r in Rr); wag = sum(r[1] for r in Rr); hands = per // 2 * W
            res[name] = dict(edge=edge.tolist(), ramp=ramp.tolist(), per_hand=tot / hands, per_unit=tot / wag)
            print(f'{name:14s}  edge TC-2 {edge[8] * 100:+.2f}%  TC0 {edge[10] * 100:+.2f}%  TC+2 {edge[12] * 100:+.2f}%  TC+4 {edge[14] * 100:+.2f}%   '
                  f'with its ramp: {tot / hands * 100:+.3f}% of a unit per hand', flush=True)
    np.save(os.path.join(out, 'count_weights.npy'), w); np.save(os.path.join(out, 'count_ramp.npy'), np.array(res['LEARNED']['ramp']))
    json.dump(dict(weights=w.tolist(), beta=beta.tolist(), results=res, rounds=a.rounds, secs=time.time() - t0), open(os.path.join(out, 'count_log.json'), 'w'))
    print(f'done in {time.time() - t0:.0f}s')
