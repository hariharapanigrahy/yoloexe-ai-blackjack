"""Casino blackjack, compiled with numba (millions of hands per minute).
Rules: 6 decks, 75% penetration, dealer stands on all 17 (S17), blackjack pays 3:2, dealer peeks for blackjack,
double on any two cards, double after split, split up to 4 hands, split aces get one card each, no surrender, no insurance.
Cards are values 2..11 (10 = ten/J/Q/K, 11 = ace). Hi-Lo count: 2-6 = +1, 10/A = -1.

Policy = int8 table pol[kind, row, dealer] with dealer up-card 2..11:
  kind 0 HARD  row = total 4..21     action 0 STAND · 1 HIT · 2 DOUBLE
  kind 1 SOFT  row = total 12..21    action 0 STAND · 1 HIT · 2 DOUBLE
  kind 2 PAIR  row = card 2..11      action 0 don't split · 1 SPLIT
Learning stats Qs/Qn[kind, row, dealer, action] collect the result of every hand that took that action in that spot."""
import numpy as np
from numba import njit

DECKS = 6; NCARDS = 52 * DECKS; CUT = int(NCARDS * 0.75)
STAND, HIT, DOUBLE, SPLIT = 0, 1, 2, 3
HILO = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1], np.int64)       # index by card value


def new_policy_random(seed=0):
    r = np.random.default_rng(seed); pol = np.zeros((3, 22, 12), np.int8)
    pol[0] = r.integers(0, 3, (22, 12)); pol[1] = r.integers(0, 3, (22, 12)); pol[2] = r.integers(0, 2, (22, 12))
    return pol


@njit(cache=True)
def make_shoe():
    s = np.empty(NCARDS, np.int64); k = 0
    for d in range(DECKS * 4):
        for v in range(2, 10): s[k] = v; k += 1
        for _ in range(4): s[k] = 10; k += 1
        s[k] = 11; k += 1
    return s


@njit(cache=True)
def add(total, soft, c):
    total += c
    if c == 11: soft += 1
    while total > 21 and soft > 0: total -= 10; soft -= 1
    return total, soft


@njit(cache=True)
def play_round(shoe, pos, rc, pol, eps, Qs, Qn, learn, bet, h17=False):
    """one round (player may split into up to 4 hands). Returns (net units won, new pos, new running count)."""
    # deal
    p1 = shoe[pos]; p2 = shoe[pos + 1]; up = shoe[pos + 2]; hole = shoe[pos + 3]; pos += 4
    rc += HILO[p1] + HILO[p2] + HILO[up]
    dt, ds = add(0, 0, up); dt, ds = add(dt, ds, hole)
    pt, ps = add(0, 0, p1); pt, ps = add(pt, ps, p2)
    if dt == 21:                                                     # dealer peeks: blackjack
        rc += HILO[hole]
        return (0.0 if pt == 21 else -1.0 * bet), pos, rc
    if pt == 21:
        rc += HILO[hole]
        return 1.5 * bet, pos, rc
    # hands stack: first card, second card (or 0), bet multiplier
    H_t = np.zeros(4, np.int64); H_s = np.zeros(4, np.int64); H_n = np.zeros(4, np.int64); H_b = np.ones(4, np.float64)
    H_c1 = np.zeros(4, np.int64); H_c2 = np.zeros(4, np.int64); H_done = np.zeros(4, np.int64); H_sa = np.zeros(4, np.int64)
    # decision log for learning: (kind,row,dealer,action,hand) up to 24 decisions
    L = np.zeros((24, 5), np.int64); nl = 0
    nh = 1; H_t[0] = pt; H_s[0] = ps; H_n[0] = 2; H_c1[0] = p1; H_c2[0] = p2
    # split decision on the initial pair (and re-splits)
    h = 0
    while h < nh:
        t = H_t[h]; s = H_s[h]
        # pair check (2 cards, same value) -> split?
        while H_n[h] == 2 and nh < 4 and H_c1[h] == H_c2[h] and H_sa[h] == 0:
            c = H_c1[h]
            a = pol[2, c, up]
            if np.random.random() < eps: a = 1 if np.random.random() < 0.5 else 0
            if learn and nl < 24: L[nl, 0] = 2; L[nl, 1] = c; L[nl, 2] = up; L[nl, 3] = a; L[nl, 4] = h; nl += 1
            if a == 0: break
            # split: this hand keeps c + new card, new hand gets c + new card
            n1 = shoe[pos]; n2 = shoe[pos + 1]; pos += 2; rc += HILO[n1] + HILO[n2]
            H_t[nh], H_s[nh] = add(0, 0, c); H_t[nh], H_s[nh] = add(H_t[nh], H_s[nh], n2); H_n[nh] = 2; H_c1[nh] = c; H_c2[nh] = n2; H_b[nh] = H_b[h]
            H_t[h], H_s[h] = add(0, 0, c); H_t[h], H_s[h] = add(H_t[h], H_s[h], n1); H_n[h] = 2; H_c2[h] = n1
            if c == 11: H_sa[h] = 1; H_sa[nh] = 1; H_done[h] = 1; H_done[nh] = 1
            nh += 1; t = H_t[h]; s = H_s[h]
        # play this hand
        while H_done[h] == 0:
            t = H_t[h]; s = H_s[h]
            if t >= 21: break
            if s > 0: kind = 1; row = t
            else: kind = 0; row = t
            a = pol[kind, row, up]
            if np.random.random() < eps: a = np.random.randint(0, 3)
            if a == DOUBLE and H_n[h] != 2: a = HIT
            if learn and nl < 24: L[nl, 0] = kind; L[nl, 1] = row; L[nl, 2] = up; L[nl, 3] = a; L[nl, 4] = h; nl += 1
            if a == STAND: break
            c = shoe[pos]; pos += 1; rc += HILO[c]
            H_t[h], H_s[h] = add(H_t[h], H_s[h], c); H_n[h] += 1
            if a == DOUBLE: H_b[h] *= 2; break
        h += 1
    # dealer
    rc += HILO[hole]
    any_live = False
    for i in range(nh):
        if H_t[i] <= 21: any_live = True
    if any_live:
        while dt < 17 or (h17 and dt == 17 and ds > 0):                # h17: dealer also hits soft 17 (claw21.com)
            c = shoe[pos]; pos += 1; rc += HILO[c]; dt, ds = add(dt, ds, c)
    # settle
    res = np.zeros(4, np.float64); net = 0.0
    for i in range(nh):
        t = H_t[i]
        if t > 21: r = -1.0
        elif dt > 21 or t > dt: r = 1.0
        elif t < dt: r = -1.0
        else: r = 0.0
        res[i] = r * H_b[i]; net += res[i]
    if learn:
        for k in range(nl):
            kind = L[k, 0]; row = L[k, 1]; dd = L[k, 2]; a = L[k, 3]; hi = L[k, 4]
            val = net if kind == 2 else res[hi]                               # a split decision is judged by the whole round
            aa = a if kind != 2 else a                                        # pair table: action 0/1
            Qs[kind, row, dd, aa] += val; Qn[kind, row, dd, aa] += 1
    return net * bet, pos, rc


@njit(cache=True)
def run_hands(n, pol, eps, seed, learn, Qs, Qn, ramp, tc_hist_ev, tc_hist_n):
    """play n rounds. ramp[tc+10] = bet in units for each true count bucket (flat 1 if ramp[0] < 0).
    Returns (total net units, total units bet). Fills per-true-count EV stats."""
    np.random.seed(seed); shoe = make_shoe(); np.random.shuffle(shoe); pos = 0; rc = 0
    tot = 0.0; wag = 0.0
    for i in range(n):
        if pos > CUT:
            np.random.shuffle(shoe); pos = 0; rc = 0
        decks_left = (NCARDS - pos) / 52.0; tc = int(np.floor(rc / max(decks_left, 0.5)))
        tci = min(20, max(0, tc + 10))
        bet = 1.0 if ramp[0] < 0 else ramp[tci]
        r, pos, rc = play_round(shoe, pos, rc, pol, eps, Qs, Qn, learn, bet)
        tot += r; wag += bet
        tc_hist_ev[tci] += r / bet; tc_hist_n[tci] += 1
    return tot, wag


def improve(pol, Qs, Qn, min_n=200):
    """the learning step: in every spot, switch to the action with the best average result so far."""
    new = pol.copy(); changed = 0
    for kind, rows in ((0, range(4, 22)), (1, range(12, 22)), (2, range(2, 12))):
        nact = 2 if kind == 2 else 3
        for r in rows:
            for d in range(2, 12):
                n = Qn[kind, r, d, :nact]
                if n.min() < min_n: continue
                q = Qs[kind, r, d, :nact] / n
                b = int(np.argmax(q))
                if b != new[kind, r, d]: new[kind, r, d] = b; changed += 1
    return new, changed
