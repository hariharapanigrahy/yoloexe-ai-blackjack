"""AI learns blackjack from zero, then learns to count cards.
  python train.py [--gens 40] [--hands 2000000] [--workers 10]
Phase 1 (STRATEGY): starts with a random chart. Each generation every worker plays `hands` rounds (10% random moves to keep
exploring), every hit/stand/double/split decision is scored by how that hand ended, then the chart switches to the best action
in every spot. Phase 2 (COUNTING): with the learned chart it plays millions of rounds while tracking the Hi-Lo true count, learns
the edge at each count, and builds a bet ramp: bet big only when its own data says the shoe favours the player.
Everything is logged to runs/<run>/log.jsonl (one line per generation: EV, chart, agreement with the textbook chart, bankroll) for the stream replay."""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bj

def book():
    exec(open(os.path.join(HERE, 'bookref.py')).read().split('Q = np.zeros')[0].split('import bj')[1], g := {'np': np})
    return g['book']()

def _work(args):
    pol, eps, seed, n, learn, ramp = args
    Q = np.zeros((3, 22, 12, 3)); N = np.zeros((3, 22, 12, 3)); ev = np.zeros(21); en = np.zeros(21)
    tot, wag = bj.run_hands(n, pol, eps, seed, learn, Q, N, ramp, ev, en)
    return tot, wag, Q, N, ev, en

def batch(ex, pol, eps, n, workers, learn, ramp, seed0):
    flat = -np.ones(21) if ramp is None else ramp
    R = list(ex.map(_work, [(pol, eps, seed0 + w, n, learn, flat) for w in range(workers)]))
    return (sum(r[0] for r in R), sum(r[1] for r in R), sum(r[2] for r in R), sum(r[3] for r in R), sum(r[4] for r in R), sum(r[5] for r in R))

def agree(pol, ref):
    tot = same = 0
    for kind, rows in ((0, range(4, 22)), (1, range(13, 22)), (2, range(2, 12))):
        for r in rows:
            for d in range(2, 12): tot += 1; same += pol[kind, r, d] == ref[kind, r, d]
    return same / tot

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='bj1'); ap.add_argument('--gens', type=int, default=40)
    ap.add_argument('--hands', type=int, default=2_000_000); ap.add_argument('--workers', type=int, default=10); a = ap.parse_args()
    out = os.path.join(HERE, 'runs', a.run); os.makedirs(out, exist_ok=True); log = open(os.path.join(out, 'log.jsonl'), 'w')
    ref = book(); pol = bj.new_policy_random(7); Qs = np.zeros((3, 22, 12, 3)); Qn = np.zeros((3, 22, 12, 3)); total_hands = 0; t0 = time.time()
    def write(rec): log.write(json.dumps(rec) + '\n'); log.flush()
    with ProcessPoolExecutor(a.workers) as ex:
        # ---------------- phase 1: learn the strategy chart ----------------
        for g in range(a.gens + 1):
            tot, wag, _, _, _, _ = batch(ex, pol, 0.0, a.hands // 2, a.workers, False, None, 10_000 + g * 97)   # how good is the chart right now
            ev = tot / wag; total_hands += a.hands // 2 * a.workers
            write({'phase': 'strategy', 'gen': g, 'hands': total_hands, 'ev': ev, 'agree': agree(pol, ref), 'chart': pol.tolist(), 'secs': time.time() - t0})
            print(f'gen {g:3d}  hands {total_hands / 1e6:8.0f}M  EV {ev * 100:+7.3f}%  matches textbook {agree(pol, ref) * 100:5.1f}%', flush=True)
            if g == a.gens: break
            eps = max(0.05, 0.5 * 0.85 ** g)
            _, _, Q, N, _, _ = batch(ex, pol, eps, a.hands, a.workers, True, None, g * 1000)
            Qs = Qs * 0.6 + Q; Qn = Qn * 0.6 + N; total_hands += a.hands * a.workers                            # newer games weigh more
            pol, changed = bj.improve(pol, Qs, Qn)
        np.save(os.path.join(out, 'chart.npy'), pol)
        # ---------------- phase 2: learn what the count is worth ----------------
        evs = np.zeros(21); ens = np.zeros(21)
        for k in range(10):
            _, _, _, _, ev_, en_ = batch(ex, pol, 0.0, 5_000_000, a.workers, False, None, 77_000 + k * 31); evs += ev_; ens += en_; total_hands += 5_000_000 * a.workers
            edge = np.where(ens > 0, evs / np.maximum(ens, 1), 0.0)
            write({'phase': 'count', 'step': k, 'hands': total_hands, 'edge_by_tc': edge.tolist(), 'n_by_tc': ens.tolist(), 'secs': time.time() - t0})
            print(f'count step {k}  hands {total_hands / 1e6:8.0f}M  edge at TC-2 {edge[8] * 100:+.2f}%  TC0 {edge[10] * 100:+.2f}%  TC+2 {edge[12] * 100:+.2f}%  TC+4 {edge[14] * 100:+.2f}%', flush=True)
        # bet ramp from its own data: sit out (watch, tiny 0.0001 bet) when the edge is below -1%, 1 unit when <= 0,
        # then about 2 units per +0.45% edge, capped at 12 units (1-12 spread + wonging: +1.6% per hand in ramps.py)
        ramp = np.array([0.0001 if e < -0.01 else 1.0 if e <= 0 else float(min(12, max(1, round(e / 0.0022)))) for e in edge])
        flat_tot, flat_wag, *_ = batch(ex, pol, 0.0, 5_000_000, a.workers, False, None, 99_000)
        cnt_tot, cnt_wag, *_ = batch(ex, pol, 0.0, 5_000_000, a.workers, False, ramp, 99_000)
        total_hands += 10_000_000 * a.workers
        write({'phase': 'final', 'hands': total_hands, 'ramp': ramp.tolist(), 'flat_ev': flat_tot / flat_wag, 'count_ev_per_hand_units': cnt_tot / (5_000_000 * a.workers),
               'count_ev_per_unit_bet': cnt_tot / cnt_wag, 'secs': time.time() - t0})
        np.save(os.path.join(out, 'ramp.npy'), ramp)
        print(f'FINAL  flat betting {flat_tot / flat_wag * 100:+.3f}%  |  counting: {cnt_tot / (5_000_000 * a.workers) * 100:+.3f}% of a unit per hand ({cnt_tot / cnt_wag * 100:+.3f}% per unit bet)  |  ramp {ramp[8:17].tolist()}', flush=True)
