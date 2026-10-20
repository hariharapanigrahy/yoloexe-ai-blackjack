import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bj
# reference "book" basic strategy (6D S17 DAS) to sanity-check the engine
def book():
    p = np.zeros((3, 22, 12), np.int8)
    for t in range(4, 22):
        for d in range(2, 12):
            if t >= 17: a = 0
            elif t >= 13: a = 0 if d <= 6 else 1
            elif t == 12: a = 0 if 4 <= d <= 6 else 1
            elif t == 11: a = 2 if d <= 10 else 1
            elif t == 10: a = 2 if d <= 9 else 1
            elif t == 9: a = 2 if 3 <= d <= 6 else 1
            else: a = 1
            p[0, t, d] = a
    for t in range(12, 22):
        for d in range(2, 12):
            if t >= 20: a = 0
            elif t == 19: a = 2 if d == 6 else 0
            elif t == 18: a = 2 if 2 <= d <= 6 else (0 if d <= 8 else 1)
            elif t == 17: a = 2 if 3 <= d <= 6 else 1
            elif t in (15, 16): a = 2 if 4 <= d <= 6 else 1
            else: a = 2 if 5 <= d <= 6 else 1
            p[1, t, d] = a
    for c in range(2, 12):
        for d in range(2, 12):
            sp = {11: True, 8: True, 10: False, 5: False, 9: d not in (7, 10, 11), 7: d <= 7, 6: d <= 6, 4: d in (5, 6), 3: d <= 7, 2: d <= 7}[c]
            p[2, c, d] = 1 if sp else 0
    return p
Q = np.zeros((3, 22, 12, 3)); N = np.zeros((3, 22, 12, 3)); ramp = -np.ones(21); ev = np.zeros(21); en = np.zeros(21)
bj.run_hands(1000, book(), 0.0, 1, False, Q, N, ramp, ev, en)
t = time.time(); n = 2_000_000
tot, wag = bj.run_hands(n, book(), 0.0, 2, False, Q, N, ramp, ev, en)
dt = time.time() - t; print(f'book strategy EV {tot / wag * 100:+.3f}%  |  {n / dt / 1e6:.2f}M hands/s on one core')
tot, wag = bj.run_hands(n, bj.new_policy_random(), 0.0, 3, False, Q, N, ramp, ev, en)
print(f'random strategy EV {tot / wag * 100:+.2f}%')
