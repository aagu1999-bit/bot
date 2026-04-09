"""
Flask Web Server — Dashboard API + Bot Control
"""

import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from bot_engine import TradingBot, BotConfig, InstrumentConfig

app = Flask(__name__, static_folder="static")
CORS(app)

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────

bot_config = BotConfig()
bot: TradingBot = None
bot_thread: threading.Thread = None
KEY_LEVELS_FILE = "key_levels.json"
CONFIG_FILE = "bot_config.json"


def load_key_levels():
    """Load key levels from disk."""
    if os.path.exists(KEY_LEVELS_FILE):
        with open(KEY_LEVELS_FILE, "r") as f:
            bot_config.key_levels = json.load(f)


def save_key_levels():
    """Persist key levels to disk."""
    with open(KEY_LEVELS_FILE, "w") as f:
        json.dump(bot_config.key_levels, f, indent=2)


def load_config():
    """Load bot config from disk."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            bot_config.environment = data.get("environment", bot_config.environment)
            bot_config.username = data.get("username", "")
            bot_config.password = data.get("password", "")
            bot_config.server = data.get("server", "")
            bot_config.max_trades_per_day = data.get("max_trades_per_day", 5)
            for name, icfg in data.get("instruments", {}).items():
                if name in bot_config.instruments:
                    bot_config.instruments[name].lot_size = icfg.get("lot_size", 0.01)
                    bot_config.instruments[name].bb_extension_pips = icfg.get("bb_extension_pips", 150)
                    bot_config.instruments[name].key_level_proximity_pips = icfg.get("key_level_proximity_pips", 50)
                    bot_config.instruments[name].stop_loss_dollars = icfg.get("stop_loss_dollars", 20)


def save_config():
    """Persist config to disk (excluding password in plaintext for production — use env vars)."""
    data = {
        "environment": bot_config.environment,
        "username": bot_config.username,
        "password": bot_config.password,  # In production, use env vars instead
        "server": bot_config.server,
        "max_trades_per_day": bot_config.max_trades_per_day,
        "instruments": {
            name: {
                "lot_size": icfg.lot_size,
                "bb_extension_pips": icfg.bb_extension_pips,
                "key_level_proximity_pips": icfg.key_level_proximity_pips,
                "stop_loss_dollars": icfg.stop_loss_dollars,
            }
            for name, icfg in bot_config.instruments.items()
        },
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ─── ROUTES: DASHBOARD ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/dashboard")
def dashboard():
    """Return full dashboard data."""
    if bot:
        # Refresh balance on every poll so the UI stays current (UI polls every 5s)
        bot.trade_manager.refresh_account_balance()
        return jsonify(bot.get_dashboard_data())
    return jsonify({
        "running": False,
        "session_active": False,
        "scan_results": {},
        "trade_status": {"connected": False, "trades_today": 0, "max_trades": bot_config.max_trades_per_day, "active_trades": [], "trade_history": [], "account_balance": None},
        "config": {
            "instruments": {k: {"symbol": v.symbol, "lot_size": v.lot_size} for k, v in bot_config.instruments.items()},
            "max_trades_per_day": bot_config.max_trades_per_day,
            "session": "23H Active · Off 3–4AM ET",
        },
        "key_levels": bot_config.key_levels,
    })


# ─── ROUTES: BOT CONTROL ─────────────────────────────────────────────────────

@app.route("/api/bot/start", methods=["POST"])
def start_bot():
    global bot, bot_thread
    if bot and bot.running:
        return jsonify({"status": "already_running"})

    bot = TradingBot(bot_config)
    bot_thread = threading.Thread(target=bot.start, daemon=True)
    bot_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/bot/stop", methods=["POST"])
def stop_bot():
    global bot
    if bot and bot.running:
        bot.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})


# ─── ROUTES: CONFIGURATION ───────────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "environment": bot_config.environment,
        "username": bot_config.username,
        "server": bot_config.server,
        "max_trades_per_day": bot_config.max_trades_per_day,
        "instruments": {
            name: {
                "symbol": icfg.symbol,
                "lot_size": icfg.lot_size,
                "bb_extension_pips": icfg.bb_extension_pips,
                "key_level_proximity_pips": icfg.key_level_proximity_pips,
                "stop_loss_dollars": icfg.stop_loss_dollars,
            }
            for name, icfg in bot_config.instruments.items()
        },
    })


@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.json
    if "environment" in data:
        bot_config.environment = data["environment"]
    if "username" in data:
        bot_config.username = data["username"]
    if "password" in data:
        bot_config.password = data["password"]
    if "server" in data:
        bot_config.server = data["server"]
    if "max_trades_per_day" in data:
        bot_config.max_trades_per_day = int(data["max_trades_per_day"])
    if "instruments" in data:
        for name, vals in data["instruments"].items():
            if name in bot_config.instruments:
                for k, v in vals.items():
                    if hasattr(bot_config.instruments[name], k):
                        setattr(bot_config.instruments[name], k, v)
    save_config()
    return jsonify({"status": "updated"})


# ─── ROUTES: KEY LEVELS ──────────────────────────────────────────────────────

@app.route("/api/key-levels", methods=["GET"])
def get_key_levels():
    return jsonify(bot_config.key_levels)


@app.route("/api/key-levels", methods=["POST"])
def update_key_levels():
    """Expects: { "NAS100": { "support": [18000, 17500], "resistance": [19000, 19500] }, ... }"""
    data = request.json
    for instrument, levels in data.items():
        if instrument in bot_config.key_levels:
            bot_config.key_levels[instrument] = levels
    save_key_levels()
    # Update bot config if bot is running
    if bot:
        bot.config.key_levels = bot_config.key_levels
    return jsonify({"status": "updated", "key_levels": bot_config.key_levels})


@app.route("/api/key-levels/<instrument>", methods=["POST"])
def add_key_level(instrument):
    """Add a single level. Expects: { "type": "support"|"resistance", "price": 18500.0 }"""
    if instrument not in bot_config.key_levels:
        return jsonify({"error": f"Unknown instrument: {instrument}"}), 400
    data = request.json
    level_type = data.get("type")
    price = float(data.get("price", 0))
    if level_type in ("support", "resistance") and price > 0:
        bot_config.key_levels[instrument][level_type].append(price)
        bot_config.key_levels[instrument][level_type].sort()
        save_key_levels()
        if bot:
            bot.config.key_levels = bot_config.key_levels
        return jsonify({"status": "added", "key_levels": bot_config.key_levels[instrument]})
    return jsonify({"error": "Invalid type or price"}), 400


@app.route("/api/key-levels/<instrument>/<level_type>/<int:index>", methods=["DELETE"])
def delete_key_level(instrument, level_type, index):
    """Delete a key level by index."""
    if instrument in bot_config.key_levels and level_type in bot_config.key_levels[instrument]:
        levels = bot_config.key_levels[instrument][level_type]
        if 0 <= index < len(levels):
            levels.pop(index)
            save_key_levels()
            if bot:
                bot.config.key_levels = bot_config.key_levels
            return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404


# ─── STARTUP ──────────────────────────────────────────────────────────────────

load_config()
load_key_levels()

if __name__ == "__main__":
    # Use environment variables for credentials in production
    bot_config.username = os.environ.get("TL_USERNAME", bot_config.username)
    bot_config.password = os.environ.get("TL_PASSWORD", bot_config.password)
    bot_config.server = os.environ.get("TL_SERVER", bot_config.server)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
