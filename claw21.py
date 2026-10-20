"""Our blackjack AI on claw21.com (online blackjack table for AI agents, free play chips, humans spectate).
Uses the chart it learned (runs/<run>/chart.npy) + Hi-Lo count over every card it sees at the table + its learned bet ramp.
claw21 rules: 6 decks, reshuffle at 25% left, dealer hits soft 17, 3:2, min bet 10, start 1000 chips. 15 s to act.
The API key is read from ~/.config/claw21/key (never printed). Every event goes to runs/claw21/log.jsonl for the stream.
  python claw21.py [--run bj2] [--unit 10] [--rounds N]"""
import os, sys, json, time, math, argparse, requests, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); API = 'https://claw21.com/api'
VAL = {'J': 10, 'Q': 10, 'K': 10, 'A': 11}; HILO = lambda v: 1 if v <= 6 else -1 if v >= 10 else 0

def val(c): return VAL.get(c['rank'], None) or int(c['rank'])
def total(cards):
    t = sum(val(c) for c in cards); a = sum(val(c) == 11 for c in cards)
    while t > 21 and a: t -= 10; a -= 1
    return t, a > 0

class Bot:
    def __init__(self, run, unit):
        self.key = open(os.path.expanduser('~/.config/claw21/key')).read().strip()
        self.pol = np.load(os.path.join(HERE, 'runs', run, 'chart.npy')); self.ramp = np.array(json.loads(open(os.path.join(HERE, 'runs', run, 'log.jsonl')).read().splitlines()[-1])['ramp'])
        self.w = np.load(os.path.join(HERE, 'runs', run, 'count_weights.npy')); self.ramp = np.load(os.path.join(HERE, 'runs', run, 'count_ramp.npy'))   # card values + ramp it discovered itself (learn_count.py)
        self.unit = unit; self.rc = 0; self.synced = False; self.counted_at = None; self.seen = set(); self.last_remaining = None; self.room = None; self.me = None; self.round_id = 0; self.bet_round = -1
        os.makedirs(os.path.join(HERE, 'runs', 'claw21'), exist_ok=True); self.logf = open(os.path.join(HERE, 'runs', 'claw21', 'log.jsonl'), 'a')
    def log(self, **k): k['t'] = time.time(); self.logf.write(json.dumps(k) + '\n'); self.logf.flush(); print(json.dumps(k)[:200], flush=True)
    def req(self, m, path, **kw):
        r = None
        for _ in range(20):
            try: r = requests.request(m, API + path, headers={'Authorization': 'Bearer ' + self.key}, timeout=15, **kw)
            except requests.RequestException: time.sleep(3); continue                # network blip: wait and retry
            if r.status_code == 429 or r.status_code >= 500: time.sleep(5); continue
            return r
        return r
    # ---------------- counting: every card that becomes visible, once per shoe ----------------
    def count(self, st):
        """count each round ONCE, when it settles and every card at the table is face up (ours, the dealer's, every other bot's).
        The bet only needs the count before the next round, so that is enough. Checked against how far the shoe went down."""
        rem = st.get('shoe', {}).get('remaining')
        if rem is None: return
        if self.last_remaining is not None and rem > self.last_remaining + 4:
            self.rc = 0; self.synced = True; self.counted_at = None; self.log(ev='shuffle')
        self.last_remaining = rem
        if st.get('phase') != 'settling' or self.counted_at == rem: return
        cards = list(st.get('dealer', {}).get('cards', []))
        for p in st.get('players', []):
            for H in p.get('hands', []): cards += H.get('cards', [])
        rc0 = self.rc; self.rc += sum(float(self.w[val(c)]) for c in cards)
        shoe_drop = (self.counted_at - rem) if self.counted_at is not None else None; self.counted_at = rem
        self.log(ev='count', cards=len(cards), shoe_drop=shoe_drop, rc=self.rc, delta=self.rc - rc0)
    def tc(self, st):
        rem = st.get('shoe', {}).get('remaining', 312); return int(math.floor(self.rc / max(rem / 52.0, 0.5)))
    # ---------------- decisions from the learned chart ----------------
    def reload(self):
        """pick up the newest brain from forever.py (runs/live/brain.npz): a chart per count + card values + ramp"""
        p = os.path.join(HERE, 'runs', 'live', 'brain.npz')
        try:
            m = os.path.getmtime(p)
            if m == getattr(self, 'brain_t', None): return
            B = np.load(p); self.POL = B['POL']; self.w = B['w']; self.ramp = B['ramp']; self.version = int(B['version']); self.brain_hands = float(B['hands']); self.brain_t = m
            self.log(ev='brain', version=self.version, hands=self.brain_hands)
        except Exception: pass
    def chart(self):                                                    # the chart for the count right now (buckets -2..+4 like forever.py)
        if getattr(self, 'POL', None) is None: return self.pol
        return self.POL[min(6, max(0, int(getattr(self, 'tc_now', 0)) + 2))]
    def decide(self, H, up, nhands):
        cards = H['cards']; t, soft = total(cards)
        if len(cards) == 2 and val(cards[0]) == val(cards[1]) and nhands < 4 and self.chart()[2, val(cards[0]), up]: return 'split'
        if t >= 21: return 'stand'                                      # 21: always stand (training auto-stands here, so the chart's 21 row is never learned)
        a = int(self.chart()[1 if soft else 0, t, up])
        if a == 2 and len(cards) != 2: a = 1
        return ['stand', 'hit', 'double'][a]
    def snapshot(self, st, ph):
        """the latest table + our numbers for the stream screen (runs/claw21/state.json)"""
        snap = dict(t=time.time(), me=self.me, state=st, rc=round(self.rc, 2), weights=self.w.tolist(), version=getattr(self, 'version', 0), brain_hands=getattr(self, 'brain_hands', 1.8e9), tc=self.tc(st), unit=self.unit, ramp=self.ramp.tolist(), last=getattr(self, 'last_dec', None),
                    leaderboard=getattr(self, 'lb', []))
        p = os.path.join(HERE, 'runs', 'claw21', 'state.json'); open(p + '.tmp', 'w').write(json.dumps(snap)); os.replace(p + '.tmp', p)
    def play(self, rounds):
        r = self.req('GET', '/me').json(); self.me = r['agentId']; self.room = r.get('currentRoomId')
        if not self.room: j = self.req('POST', '/join').json(); self.room = j['roomId']; self.log(ev='join', **j)
        last_phase = None; done = 0
        while done < rounds:
            st = self.req('GET', f'/state?room={self.room}')
            if st is None or st.status_code >= 500: time.sleep(5); continue
            if st.status_code == 404: j = self.req('POST', '/join').json(); self.room = j['roomId']; self.log(ev='rejoin', **j); continue
            st = st.json(); ph = st.get('phase'); self.count(st)
            mine = next((p for p in st.get('players', []) if p['agentId'] == self.me), None)
            if time.time() - getattr(self, '_lb_t', 0) > 20:                 # public leaderboard for the stream, every 2 min
                self._lb_t = time.time()
                try: self.lb = requests.get(API + '/leaderboard', timeout=10).json().get('leaderboard', [])
                except Exception: pass
            self.snapshot(st, ph)
            if ph != last_phase and ph == 'betting': self.round_id += 1
            if ph == 'betting' and self.bet_round != self.round_id and mine:
                self.reload(); tc = self.tc(st); self.tc_now = tc; u = self.ramp[min(20, max(0, tc + 10))] if self.synced else 1.0; self.bet_round = self.round_id   # joined mid-shoe: flat min bet until the next shuffle
                if u < 0.01: self.log(ev='sit_out', tc=tc, rc=self.rc, chips=mine['chips'])
                else:
                    amt = int(min(mine['chips'], max(10, round(u * self.unit))))
                    r = self.req('POST', '/bet', json={'roomId': self.room, 'amount': amt}); self.log(ev='bet', amount=amt, tc=tc, rc=self.rc, chips=mine['chips'], ok=r.status_code)
            elif ph == 'player_turns' and mine and mine.get('isActive'):
                H = mine['hands'][mine.get('currentHandIndex', 0)]; up = val(st['dealer']['cards'][0]); a = self.decide(H, up, len(mine['hands']))
                r = self.req('POST', '/action', json={'roomId': self.room, 'action': a})
                if r.status_code == 400 and a in ('split', 'double'):              # table refused (rule differs): fall back to hit/stand by the chart
                    t, soft = total(H['cards']); b = int(self.chart()[1 if soft else 0, t, up]); a2 = 'hit' if b else 'stand'
                    r = self.req('POST', '/action', json={'roomId': self.room, 'action': a2}); a = f'{a}->{a2}'
                self.last_dec = dict(action=a, hand=[c['rank'] for c in H['cards']], up=up, t=time.time())
                self.log(ev='action', action=a, hand=[c['rank'] for c in H['cards']], up=up, tc=self.tc(st), ok=r.status_code); time.sleep(0.4); continue
            elif ph == 'settling' and last_phase != 'settling':
                res = [s for s in st.get('settlements', []) if s['agentId'] == self.me]
                if res: done += 1
                self.log(ev='settle', results=res, dealer=[c['rank'] for c in st['dealer']['cards']], chips=(mine or {}).get('chips'), tc=self.tc(st), rc=self.rc,
                         players=len(st.get('players', [])), round=done)
            last_phase = ph; time.sleep(1.3)                                     # 60 GETs per minute limit

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='bj2'); ap.add_argument('--unit', type=int, default=10); ap.add_argument('--rounds', type=int, default=10 ** 9)
    a = ap.parse_args(); Bot(a.run, a.unit).play(a.rounds)
