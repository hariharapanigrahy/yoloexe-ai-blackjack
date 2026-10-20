"""Never-ending training in the background, on claw21.com's exact rules (6 decks, 75% dealt, dealer HITS soft 17, 3:2).
Every generation the AI:
  1. plays millions of hands with a separate strategy chart for each count (-2 or less, -1, 0, +1, +2, +3, +4 or more): it learns
     that some plays change when the deck is hot or cold (e.g. stand on 16 vs 10 when the count is high);
  2. every few generations re-learns its own card values and what each count is worth (bet ramp);
  3. a new brain is only published if it beats the current one on the same 30M hands (no getting worse).
Published brain: runs/live/brain.npz (charts per count, card values, ramp, version) -> claw21.py and the stream pick it up live.
  python forever.py [--workers 6]"""
import os, sys, json, time, argparse, numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bj, learn_count as LC
OUT = os.path.join(HERE, 'runs', 'live'); NB = 7                     # count buckets: <=-2, -1, 0, +1, +2, +3, >=+4

@njit(cache=True)
def bucket(tci): return min(NB - 1, max(0, tci + 2))

@njit(cache=True)
def run_tc(n, POL, eps, seed, learn, QS, QN, w, ramp, ev, en):
    np.random.seed(seed); shoe = bj.make_shoe(); np.random.shuffle(shoe); pos = 0; rc = 0; run = 0.0; tot = 0.0; wag = 0.0
    for i in range(n):
        if pos > bj.CUT: np.random.shuffle(shoe); pos = 0; run = 0.0
        tci = int(np.floor(run / max((bj.NCARDS - pos) / 52.0, 0.5))); b = bucket(tci); ti = min(20, max(0, tci + 10))
        bet = 1.0 if (learn or ramp[0] < 0) else ramp[ti]
        r, p2, rc = bj.play_round(shoe, pos, rc, POL[b], eps, QS[b], QN[b], learn, bet, True)
        for k in range(pos, p2): run += w[shoe[k]]
        pos = p2; tot += r; wag += bet; ev[ti] += r / bet; en[ti] += 1
    return tot, wag

def _run(a):
    POL, eps, seed, n, learn, w, ramp = a; QS = np.zeros((NB, 3, 22, 12, 3)); QN = np.zeros_like(QS); ev = np.zeros(21); en = np.zeros(21)
    t, g = run_tc(n, POL, eps, seed, learn, QS, QN, w, ramp, ev, en); return t, g, QS, QN, ev, en
def batch(ex, W, POL, eps, n, learn, w, ramp, seed0):
    R = list(ex.map(_run, [(POL, eps, seed0 + k, n, learn, w, ramp) for k in range(W)]))
    return [sum(r[i] for r in R) for i in range(6)]
def evaluate(ex, W, B, n, seed):
    """per-hand result (units) of a brain with its own ramp, on fixed seeds so two brains see the same cards"""
    t, g, _, _, _, _ = batch(ex, W, B['POL'], 0.0, n, False, B['w'], B['ramp'], seed); return t / (n * W)
def relearn_count(ex, W, B, n):
    """card values (effect of removal, H17 rules, chart for count 0) + edge per count + ramp"""
    R = list(ex.map(LC._gather, [(B['POL'][2], 70_000 + k, n, True) for k in range(W)]))
    XtX = sum(r[0] for r in R); Xty = sum(r[1] for r in R); beta = np.linalg.solve(XtX + 1e-6 * np.eye(11), Xty)
    eor = beta[:10] - (beta[:10] * LC.PFULL[2:]).sum(); w = np.zeros(12); w[2:] = eor / -eor[8]
    _, _, _, _, ev, en = batch(ex, W, B['POL'], 0.0, n, False, w, -np.ones(21), 80_000)
    edge = np.where(en > 0, ev / np.maximum(en, 1), 0.0); return w, LC.ramp_from(edge), edge
def save(B):
    os.makedirs(OUT, exist_ok=True); np.savez(os.path.join(OUT, 'brain_tmp.npz'), **{k: np.asarray(v) for k, v in B.items()})
    os.replace(os.path.join(OUT, 'brain_tmp.npz'), os.path.join(OUT, 'brain.npz'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--hands', type=int, default=3_000_000)
    ap.add_argument('--eval', type=int, default=5_000_000); a = ap.parse_args(); W = a.workers; os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, 'log.jsonl'), 'a'); t0 = time.time()
    def write(**k): k['t'] = time.time(); log.write(json.dumps(k) + '\n'); log.flush(); print(json.dumps({x: y for x, y in k.items() if x not in ('edge', 'weights')})[:240], flush=True)
    p = os.path.join(OUT, 'brain.npz')
    if os.path.exists(p): B = {k: v for k, v in np.load(p).items()}; B['version'] = int(B['version']); B['hands'] = float(B['hands'])
    else:                                                               # start from the brain it already has (bj2 chart + its own count)
        base = os.path.join(HERE, 'runs', 'bj2'); chart = np.load(os.path.join(base, 'chart.npy'))
        B = dict(POL=np.stack([chart] * NB), w=np.load(os.path.join(base, 'count_weights.npy')), ramp=np.load(os.path.join(base, 'count_ramp.npy')), version=1, hands=1.8e9)
        save(B)
    QS = np.zeros((NB, 3, 22, 12, 3)); QN = np.zeros_like(QS); g = 0
    with ProcessPoolExecutor(W) as ex:
        best = evaluate(ex, W, B, a.eval, 123_456); write(ev='start', version=B['version'], per_hand=best)
        while True:
            g += 1; eps = 0.08
            _, _, Q, N, _, _ = batch(ex, W, B['POL'], eps, a.hands, True, B['w'], -np.ones(21), 1_000_000 + g * 101)
            QS = QS * 0.9 + Q; QN = QN * 0.9 + N; B['hands'] += a.hands * W
            cand = dict(B); cand['POL'] = np.stack([bj.improve(B['POL'][b], QS[b], QN[b], min_n=3000)[0] for b in range(NB)])
            changed = int((cand['POL'] != B['POL']).sum())
            if g % 10 == 0:                                             # re-learn card values + ramp for the current charts
                w, ramp, edge = relearn_count(ex, W, cand, a.hands); cand['w'] = w; cand['ramp'] = ramp
            if changed or g % 10 == 0:
                seed = 123_456 + g * 7                                  # fresh shared cards for both brains every test
                old = evaluate(ex, W, B, a.eval, seed); new = evaluate(ex, W, cand, a.eval, seed)
                if new > old:
                    cand['version'] = B['version'] + 1; B = cand; save(B)
                    write(ev='published', gen=g, version=B['version'], per_hand=new, beat=old, changed=changed, hands=B['hands'], weights=B['w'].tolist(), mins=(time.time() - t0) / 60)
                else: write(ev='kept', gen=g, version=B['version'], per_hand=old, rejected=new, changed=changed, hands=B['hands'])
            else: write(ev='gen', gen=g, version=B['version'], hands=B['hands'])
