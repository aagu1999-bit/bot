"""
Trading Bot Engine — BB + EMA 9 Multi-Timeframe Strategy
Instruments: NAS100, US30, SPX500
Strategy: 4H bias → 30m confirm → 5m trigger → 1m execute
"""

import time
import json
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from tradelocker import TLAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingBot")


# ─── CONFIGURATION ───────────────────────────────────────────────────────────

@dataclass
class InstrumentConfig:
    symbol: str
    lot_size: float = 0.01
    bb_extension_pips: float = 150.0  # pips beyond BB for overextension signal
    key_level_proximity_pips: float = 50.0  # pips within key level for signal
    stop_loss_dollars: float = 20.0
    pip_value_per_lot: float = 1.0  # $ per pip per 1.0 lot — set per instrument


@dataclass
class BotConfig:
    environment: str = "https://demo.tradelocker.com"
    username: str = ""
    password: str = ""
    server: str = ""
    max_trades_per_day: int = 5
    session_start_hour_et: int = 3   # 3 AM ET (London open)
    session_end_hour_et: int = 16    # 4 PM ET (NY close)
    instruments: dict = field(default_factory=lambda: {
        "NAS100": InstrumentConfig(
            symbol="NAS100",
            lot_size=0.01,
            bb_extension_pips=150.0,
            key_level_proximity_pips=50.0,
            stop_loss_dollars=20.0,
            pip_value_per_lot=0.10,  # approximate — adjust per broker
        ),
        "US30": InstrumentConfig(
            symbol="US30",
            lot_size=0.01,
            bb_extension_pips=100.0,
            key_level_proximity_pips=35.0,
            stop_loss_dollars=20.0,
            pip_value_per_lot=0.10,
        ),
        "SPX500": InstrumentConfig(
            symbol="SPX500",
            lot_size=0.01,
            bb_extension_pips=30.0,
            key_level_proximity_pips=10.0,
            stop_loss_dollars=20.0,
            pip_value_per_lot=0.10,
        ),
    })
    key_levels: dict = field(default_factory=lambda: {
        "NAS100": {"support": [], "resistance": []},
        "US30": {"support": [], "resistance": []},
        "SPX500": {"support": [], "resistance": []},
    })


# ─── INDICATOR CALCULATIONS ──────────────────────────────────────────────────

def calc_ema(closes: np.ndarray, period: int = 9) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    ema = np.zeros_like(closes)
    multiplier = 2.0 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = (closes[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema


def calc_bollinger_bands(closes: np.ndarray, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
    if len(closes) < period:
        return None, None, None
    middle = np.convolve(closes, np.ones(period) / period, mode="valid")
    # Pad the beginning
    pad_len = len(closes) - len(middle)
    middle = np.concatenate([np.full(pad_len, np.nan), middle])

    std = np.array([
        np.std(closes[max(0, i - period + 1):i + 1]) if i >= period - 1 else np.nan
        for i in range(len(closes))
    ])
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def calc_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Calculate Volume Weighted Average Price (daily reset assumed per session)."""
    typical_price = (highs + lows + closes) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volumes)
    cum_vol = np.cumsum(volumes)
    cum_vol[cum_vol == 0] = 1  # avoid division by zero
    return cum_tp_vol / cum_vol


# ─── SIGNAL LOGIC ────────────────────────────────────────────────────────────

@dataclass
class TimeframeData:
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    volumes: np.ndarray
    ema9: np.ndarray = None
    bb_upper: np.ndarray = None
    bb_middle: np.ndarray = None
    bb_lower: np.ndarray = None
    vwap: np.ndarray = None

    def compute_indicators(self):
        self.ema9 = calc_ema(self.closes, 9)
        self.bb_upper, self.bb_middle, self.bb_lower = calc_bollinger_bands(self.closes, 20, 2.0)
        if len(self.volumes) > 0 and np.sum(self.volumes) > 0:
            self.vwap = calc_vwap(self.highs, self.lows, self.closes, self.volumes)
        else:
            self.vwap = self.bb_middle  # fallback


class SignalEngine:
    """Evaluates the multi-timeframe strategy."""

    def __init__(self, config: BotConfig):
        self.config = config

    def check_4h_bias(self, tf: TimeframeData, instrument: str) -> Optional[str]:
        """
        Returns 'LONG', 'SHORT', or None.
        LONG if:
          - price within key_level_proximity of a support level, OR
          - price is bb_extension_pips below the lower BB
        SHORT if:
          - price within key_level_proximity of a resistance level, OR
          - price is bb_extension_pips above the upper BB
        """
        if tf.bb_lower is None or np.isnan(tf.bb_lower[-1]):
            return None

        price = tf.closes[-1]
        icfg = self.config.instruments[instrument]
        key = self.config.key_levels.get(instrument, {"support": [], "resistance": []})

        # Check key support levels
        for level in key["support"]:
            if abs(price - level) <= icfg.key_level_proximity_pips:
                logger.info(f"[4H] {instrument} LONG bias — price {price:.2f} near support {level}")
                return "LONG"

        # Check overextension below lower BB
        if price < tf.bb_lower[-1] - icfg.bb_extension_pips:
            logger.info(f"[4H] {instrument} LONG bias — overextended {price:.2f} below lower BB {tf.bb_lower[-1]:.2f}")
            return "LONG"

        # Check key resistance levels
        for level in key["resistance"]:
            if abs(price - level) <= icfg.key_level_proximity_pips:
                logger.info(f"[4H] {instrument} SHORT bias — price {price:.2f} near resistance {level}")
                return "SHORT"

        # Check overextension above upper BB
        if price > tf.bb_upper[-1] + icfg.bb_extension_pips:
            logger.info(f"[4H] {instrument} SHORT bias — overextended {price:.2f} above upper BB {tf.bb_upper[-1]:.2f}")
            return "SHORT"

        return None

    def check_30m_confirmation(self, tf: TimeframeData, bias: str) -> bool:
        """
        Confirmed if the last candle is respecting (within ~10 pips of)
        any of: VWAP, middle BB, upper BB, lower BB, or EMA 9.
        For LONG: candle closed bullish (green). For SHORT: candle closed bearish (red).
        """
        if tf.bb_lower is None or np.isnan(tf.bb_lower[-1]):
            return False

        price = tf.closes[-1]
        prev_close = tf.closes[-2] if len(tf.closes) > 1 else price
        candle_low = tf.lows[-1]
        candle_high = tf.highs[-1]
        proximity = 10.0  # pips

        # Check if candle is respecting any key indicator level
        reference_levels = [
            tf.ema9[-1],
            tf.bb_middle[-1],
            tf.bb_upper[-1],
            tf.bb_lower[-1],
        ]
        if tf.vwap is not None and not np.isnan(tf.vwap[-1]):
            reference_levels.append(tf.vwap[-1])

        respecting = False
        for level in reference_levels:
            if np.isnan(level):
                continue
            if bias == "LONG" and abs(candle_low - level) <= proximity and price > prev_close:
                respecting = True
                break
            if bias == "SHORT" and abs(candle_high - level) <= proximity and price < prev_close:
                respecting = True
                break

        if respecting:
            logger.info(f"[30m] Confirmation {'LONG' if bias == 'LONG' else 'SHORT'} — price respecting level")
        return respecting

    def check_5m_trigger(self, tf: TimeframeData, bias: str) -> Optional[float]:
        """
        Checks if price breaks above (LONG) or below (SHORT) the 4-candle consolidation range.
        Returns the breakout level if triggered, else None.
        """
        if len(tf.closes) < 5:
            return None

        # Last 4 candles before the current one define the consolidation range
        consol_highs = tf.highs[-5:-1]
        consol_lows = tf.lows[-5:-1]
        range_high = np.max(consol_highs)
        range_low = np.min(consol_lows)

        current_close = tf.closes[-1]

        if bias == "LONG" and current_close > range_high:
            logger.info(f"[5m] LONG trigger — broke above consolidation {range_high:.2f}")
            return range_high
        if bias == "SHORT" and current_close < range_low:
            logger.info(f"[5m] SHORT trigger — broke below consolidation {range_low:.2f}")
            return range_low

        return None

    def check_1m_confirmation(self, tf: TimeframeData, bias: str, breakout_level: float) -> bool:
        """
        Confirmed if the 1m candle closes beyond the 5m breakout level.
        """
        price = tf.closes[-1]
        if bias == "LONG" and price > breakout_level:
            logger.info(f"[1m] LONG confirmed — closed {price:.2f} above breakout {breakout_level:.2f}")
            return True
        if bias == "SHORT" and price < breakout_level:
            logger.info(f"[1m] SHORT confirmed — closed {price:.2f} below breakout {breakout_level:.2f}")
            return True
        return False


# ─── TRADE MANAGER ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    instrument: str
    side: str  # "buy" or "sell"
    entry_price: float
    lot_size: float
    stop_loss_price: float
    tp1_price: float  # middle BB on 30m
    tp2_price: float  # opposite BB or VWAP on 30m
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    status: str = "pending"  # pending, active, partial, closed
    pnl: float = 0.0
    entry_time: str = ""
    exit_time: str = ""
    tp1_hit: bool = False


class TradeManager:
    """Manages order execution and position tracking via TradeLocker API."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.tl: Optional[TLAPI] = None
        self.trades_today: list[Trade] = []
        self.trade_history: list[Trade] = []
        self.connected = False
        self.instrument_ids: dict = {}

    def connect(self) -> bool:
        """Connect to TradeLocker API."""
        try:
            self.tl = TLAPI(
                environment=self.config.environment,
                username=self.config.username,
                password=self.config.password,
                server=self.config.server,
            )
            # Resolve instrument IDs
            all_instruments = self.tl.get_all_instruments()
            for name in self.config.instruments:
                try:
                    iid = self.tl.get_instrument_id_from_symbol_name(name)
                    self.instrument_ids[name] = iid
                    logger.info(f"Resolved {name} → instrument ID {iid}")
                except Exception:
                    logger.warning(f"Could not resolve instrument: {name}")
            self.connected = True
            logger.info("Connected to TradeLocker API (demo)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TradeLocker: {e}")
            self.connected = False
            return False

    def get_candles(self, instrument: str, resolution: str, lookback: str) -> Optional[TimeframeData]:
        """Fetch candle data and compute indicators."""
        if not self.connected or instrument not in self.instrument_ids:
            return None
        try:
            data = self.tl.get_price_history(
                self.instrument_ids[instrument],
                resolution=resolution,
                start_timestamp=0,
                end_timestamp=0,
                lookback_period=lookback,
            )
            if data is None or len(data) == 0:
                return None

            # TradeLocker returns OHLCV data
            # Format: columns = [timestamp, open, high, low, close, volume]
            closes = np.array(data["c"], dtype=float)
            highs = np.array(data["h"], dtype=float)
            lows = np.array(data["l"], dtype=float)
            volumes = np.array(data["v"], dtype=float) if "v" in data else np.ones_like(closes)

            tf = TimeframeData(closes=closes, highs=highs, lows=lows, volumes=volumes)
            tf.compute_indicators()
            return tf
        except Exception as e:
            logger.error(f"Error fetching candles for {instrument} {resolution}: {e}")
            return None

    def execute_trade(self, instrument: str, side: str, lot_size: float,
                      stop_loss: float, tp1: float, tp2: float) -> Optional[Trade]:
        """Place a market order via TradeLocker."""
        if not self.connected:
            return None

        if len(self.trades_today) >= self.config.max_trades_per_day:
            logger.warning(f"Max trades per day ({self.config.max_trades_per_day}) reached")
            return None

        try:
            iid = self.instrument_ids[instrument]
            latest_price = self.tl.get_latest_asking_price(iid)

            order_id = self.tl.create_order(
                instrument_id=iid,
                quantity=lot_size,
                side=side,
                type_="market",
                stop_loss=stop_loss,
                take_profit=tp1,  # Set TP1 as initial take-profit
            )

            if order_id:
                trade = Trade(
                    instrument=instrument,
                    side=side,
                    entry_price=latest_price,
                    lot_size=lot_size,
                    stop_loss_price=stop_loss,
                    tp1_price=tp1,
                    tp2_price=tp2,
                    order_id=str(order_id),
                    status="active",
                    entry_time=datetime.now(timezone.utc).isoformat(),
                )
                self.trades_today.append(trade)
                logger.info(f"EXECUTED {side.upper()} {instrument} @ {latest_price:.2f} | SL: {stop_loss:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f}")
                return trade
            else:
                logger.error(f"Order creation failed for {instrument}")
                return None
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None

    def calculate_stop_loss(self, instrument: str, side: str, entry_price: float) -> float:
        """Calculate stop loss price based on fixed $20 risk."""
        icfg = self.config.instruments[instrument]
        # stop_loss_dollars / (lot_size * pip_value_per_lot) = pips of stop
        stop_pips = icfg.stop_loss_dollars / (icfg.lot_size * icfg.pip_value_per_lot)
        if side == "buy":
            return entry_price - stop_pips
        else:
            return entry_price + stop_pips

    def get_status(self) -> dict:
        """Return current bot status for the dashboard."""
        return {
            "connected": self.connected,
            "trades_today": len(self.trades_today),
            "max_trades": self.config.max_trades_per_day,
            "active_trades": [asdict(t) for t in self.trades_today if t.status in ("active", "partial")],
            "trade_history": [asdict(t) for t in self.trade_history[-50:]],
        }


# ─── MAIN BOT LOOP ───────────────────────────────────────────────────────────

class TradingBot:
    """Main orchestrator that runs the strategy loop."""

    # TradeLocker resolution codes
    TF_MAP = {
        "4H": ("4H", "30D"),
        "1H": ("1H", "14D"),
        "30m": ("30m", "7D"),
        "5m": ("5m", "2D"),
        "1m": ("1m", "1D"),
    }

    def __init__(self, config: BotConfig):
        self.config = config
        self.signal_engine = SignalEngine(config)
        self.trade_manager = TradeManager(config)
        self.running = False
        self.scan_results: dict = {}  # latest scan state per instrument

    def start(self):
        if not self.trade_manager.connect():
            logger.error("Cannot start — TradeLocker connection failed")
            return
        self.running = True
        logger.info("Bot started")
        self._run_loop()

    def stop(self):
        self.running = False
        logger.info("Bot stopped")

    def _is_in_session(self) -> bool:
        """Check if current time is within London+NY session (ET)."""
        now_utc = datetime.now(timezone.utc)
        # ET = UTC-4 (EDT) or UTC-5 (EST) — simplified to UTC-4 for now
        et_hour = (now_utc.hour - 4) % 24
        return self.config.session_start_hour_et <= et_hour < self.config.session_end_hour_et

    def _run_loop(self):
        """Main loop — scans every 60 seconds."""
        while self.running:
            try:
                if not self._is_in_session():
                    logger.info("Outside trading session — sleeping 60s")
                    time.sleep(60)
                    continue

                if len(self.trade_manager.trades_today) >= self.config.max_trades_per_day:
                    logger.info("Max trades reached for today — sleeping 300s")
                    time.sleep(300)
                    continue

                for instrument in self.config.instruments:
                    self._scan_instrument(instrument)

                time.sleep(60)  # Scan every 60 seconds

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(30)

    def _scan_instrument(self, instrument: str):
        """Run full MTF scan on a single instrument."""
        icfg = self.config.instruments[instrument]
        scan = {"instrument": instrument, "timestamp": datetime.now(timezone.utc).isoformat()}

        # Step 1: 4H bias
        tf_4h = self.trade_manager.get_candles(instrument, *self.TF_MAP["4H"])
        if tf_4h is None:
            scan["4h_bias"] = "NO DATA"
            self.scan_results[instrument] = scan
            return

        bias = self.signal_engine.check_4h_bias(tf_4h, instrument)
        scan["4h_bias"] = bias or "NEUTRAL"
        if bias is None:
            self.scan_results[instrument] = scan
            return

        # Step 2: 30m confirmation
        tf_30m = self.trade_manager.get_candles(instrument, *self.TF_MAP["30m"])
        if tf_30m is None:
            scan["30m_confirm"] = "NO DATA"
            self.scan_results[instrument] = scan
            return

        confirmed = self.signal_engine.check_30m_confirmation(tf_30m, bias)
        scan["30m_confirm"] = "YES" if confirmed else "NO"
        if not confirmed:
            self.scan_results[instrument] = scan
            return

        # Step 3: 5m trigger
        tf_5m = self.trade_manager.get_candles(instrument, *self.TF_MAP["5m"])
        if tf_5m is None:
            scan["5m_trigger"] = "NO DATA"
            self.scan_results[instrument] = scan
            return

        breakout_level = self.signal_engine.check_5m_trigger(tf_5m, bias)
        scan["5m_trigger"] = f"BREAKOUT @ {breakout_level:.2f}" if breakout_level else "NO"
        if breakout_level is None:
            self.scan_results[instrument] = scan
            return

        # Step 4: 1m confirmation
        tf_1m = self.trade_manager.get_candles(instrument, *self.TF_MAP["1m"])
        if tf_1m is None:
            scan["1m_confirm"] = "NO DATA"
            self.scan_results[instrument] = scan
            return

        entry_confirmed = self.signal_engine.check_1m_confirmation(tf_1m, bias, breakout_level)
        scan["1m_confirm"] = "ENTRY!" if entry_confirmed else "NO"

        if entry_confirmed:
            side = "buy" if bias == "LONG" else "sell"
            entry_price = tf_1m.closes[-1]

            # Calculate levels
            stop_loss = self.trade_manager.calculate_stop_loss(instrument, side, entry_price)
            tp1 = tf_30m.bb_middle[-1] if not np.isnan(tf_30m.bb_middle[-1]) else entry_price
            # TP2: opposite BB or VWAP — whichever is closer
            if side == "buy":
                tp2_bb = tf_30m.bb_upper[-1] if not np.isnan(tf_30m.bb_upper[-1]) else tp1
            else:
                tp2_bb = tf_30m.bb_lower[-1] if not np.isnan(tf_30m.bb_lower[-1]) else tp1

            tp2_vwap = tf_30m.vwap[-1] if (tf_30m.vwap is not None and not np.isnan(tf_30m.vwap[-1])) else tp2_bb

            # Pick the closer of opposite BB and VWAP for TP2
            if side == "buy":
                tp2 = min(tp2_bb, tp2_vwap) if tp2_vwap > entry_price else tp2_bb
            else:
                tp2 = max(tp2_bb, tp2_vwap) if tp2_vwap < entry_price else tp2_bb

            self.trade_manager.execute_trade(
                instrument=instrument,
                side=side,
                lot_size=icfg.lot_size,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
            )

        self.scan_results[instrument] = scan

    def get_dashboard_data(self) -> dict:
        """Return all data needed for the UI dashboard."""
        return {
            "running": self.running,
            "session_active": self._is_in_session(),
            "scan_results": self.scan_results,
            "trade_status": self.trade_manager.get_status(),
            "config": {
                "instruments": {k: asdict(v) for k, v in self.config.instruments.items()},
                "max_trades_per_day": self.config.max_trades_per_day,
                "session": f"{self.config.session_start_hour_et}:00 - {self.config.session_end_hour_et}:00 ET",
            },
            "key_levels": self.config.key_levels,
        }
