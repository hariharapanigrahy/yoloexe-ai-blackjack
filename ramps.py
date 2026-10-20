"""compare bet ramps on the learned chart"""
import numpy as np, sys, os
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import train as T
if __name__ == '__main__':
    pol = np.load('runs/bj2/chart.npy')
    ramps = {'1-5 (found)': np.load('runs/bj2/ramp.npy'),
             '1-12': np.array([1.0] * 11 + [2, 4, 6, 8, 10, 12, 12, 12, 12, 12], float),
             '1-12 + sit out TC<=-2': np.array([0.0001] * 9 + [1, 1] + [2, 4, 6, 8, 10, 12, 12, 12, 12, 12], float)}
    with ProcessPoolExecutor(10) as ex:
        for name, r in ramps.items():
            tot, wag, *_ = T.batch(ex, pol, 0.0, 5_000_000, 10, False, r, 4242)
            print(f'{name:24s}  per hand {tot / 50e6 * 100:+.3f}% of a unit   per unit bet {tot / wag * 100:+.3f}%', flush=True)
