# MTF Trading Bot — BB + EMA 9 Strategy

## Strategy Summary

Multi-timeframe day trading bot for indices (NAS100, US30, SPX500) using Bollinger Bands (20/2), EMA 9, and VWAP.

**Entry Logic:**
1. **4H Bias** — Price within 50 pips of a manually-set key level OR 150+ pips beyond BB (overextension)
2. **30m Confirmation** — Candle respecting VWAP/BB/EMA 9 with directional close
3. **5m Trigger** — Breakout above/below 4-candle consolidation range
4. **1m Entry** — Candle closes beyond the 5m breakout level → EXECUTE

**Exit Logic:**
- TP1 (50%): 30m middle Bollinger Band
- TP2 (50%): 30m opposite BB or VWAP (whichever hits first)
- Stop Loss: $20 fixed

**Sessions:** London + NY (3 AM – 4 PM ET)
**Max Trades/Day:** 3–5 (configurable)

---

## Setup on Replit

### 1. Create a New Repl
- Go to replit.com → Create Repl → Choose "Python"
- Upload all files from this project

### 2. Install Dependencies
In the Replit Shell, run:
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Go to **Tools → Secrets** in Replit and add:
- `TL_USERNAME` — your TradeLocker email
- `TL_PASSWORD` — your TradeLocker password
- `TL_SERVER` — your TradeLocker server name

### 4. Run the Bot
Click the Run button or run:
```bash
python server.py
```

The dashboard will be available at your Replit URL.

### 5. Configure Key Levels
- Go to the **Key Levels** tab in the dashboard
- Add your support and resistance levels for each instrument
- These are the 4H/1H levels you've drawn dating back to August 2024

### 6. Start the Bot
- Go to the **Settings** tab and verify your connection settings
- Click **Start** in the header
- The bot will begin scanning every 60 seconds during London + NY sessions

---

## File Structure

```
├── server.py           # Flask web server + API routes
├── bot_engine.py       # Trading bot engine + strategy logic
├── static/
│   └── index.html      # Dashboard UI
├── requirements.txt    # Python dependencies
├── .replit             # Replit config
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## Important Notes

- **Start with DEMO** — Always test on a demo account first
- **Key Levels** — The bot cannot auto-detect key levels. You must input them manually
- **Pip Values** — The pip_value_per_lot in bot_engine.py may need adjusting based on your broker's contract specifications
- **Rate Limits** — TradeLocker has API rate limits. The bot scans every 60 seconds to stay within limits
- **Replit Always-On** — For 24/5 operation, you may need Replit's paid plan for "Always On" or use a deployment

---

## Customization

### Adjust Lot Sizes
Settings tab → Change lot size per instrument (default: 0.01)

### Adjust BB Extension Thresholds
Settings tab → Change "BB Extension (pips)" per instrument:
- NAS100: 150 (default)
- US30: 100
- SPX500: 30

### Adjust Key Level Proximity
Settings tab → Change "Key Level Proximity (pips)" per instrument:
- NAS100: 50 (default)
- US30: 35
- SPX500: 10
