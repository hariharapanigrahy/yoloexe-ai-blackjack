# AI vs THE HOUSE 🃏

An AI that learned blackjack from **zero**. No strategy chart, no card-counting system given to it.

- Starts with a **random** brain (loses 48% per hand).
- Plays **1.8 billion hands in ~90 seconds** and teaches itself the strategy (hit / stand / double / split). It ends up 94% identical to the textbook chart humans worked out.
- Then it **discovers card counting on its own**: which cards help it when they're gone from the shoe. It finds the 5 is the most important card, and its count beats the classic human Hi-Lo system (+2.0% vs +1.8% per hand).
- Then it plays **live online** against other AI bots on [claw21.com](https://claw21.com) (free chips), and it hit **#1 on the leaderboard**.

Watch it play live: **YOLO.exe** on YouTube and Kick.

> Free chips / simulation only. This is a science project, not a way to make money. Don't bring bots to real-money casinos: it breaks their rules (and often the law), and online casinos reshuffle every hand anyway.

## Run it

```bash
pip install -r requirements.txt
python train.py --run mine          # learn the strategy from a random brain + what each count is worth (~1-2 min)
python learn_count.py --run mine    # let it discover its own card values and compare with Hi-Lo
python forever.py                   # never-ending training on claw21 rules (dealer hits soft 17), publishes runs/live/brain.npz
```

A trained brain is already included (`runs/bj2`, `runs/live/brain.npz`), so you can skip straight to playing.

### Play on claw21.com (AI-only blackjack table, free chips)
1. Register your bot (one time) and save the key:
   ```bash
   mkdir -p ~/.config/claw21 && curl -s -X POST https://claw21.com/api/register -H 'Content-Type: application/json' -d '{"name":"YOUR-BOT-NAME"}'
   ```
   Put the `apiKey` from the reply into `~/.config/claw21/key`. Keep it secret.
2. `python claw21.py`: it joins a table, counts every card it sees (yours, the dealer's, other bots'), bets by its own count and plays its learned chart.

## Files
| file | what it does |
|---|---|
| `bj.py` | the casino: 6 decks, 3:2, peek, double after split, up to 4 hands. Compiled with numba, ~4M hands/s per CPU core |
| `train.py` | learning the strategy from zero (Monte Carlo policy improvement) + edge by true count + bet ramp |
| `learn_count.py` | the AI discovers its own card-counting values (effect of removal) and compares with Hi-Lo |
| `forever.py` | nonstop training: a chart per count, re-learned card values, only publishes a brain that beats the last one |
| `claw21.py` | plays live on claw21.com |
| `bookref.py` | the human textbook chart, only used to score how close the AI got |

MIT license. Have fun. Show us your bot's leaderboard rank in the Discord: https://discord.gg/EmrQ656bj
