from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from backtesting import Backtest, Strategy


# Ichimoku Intro
# https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/Ichimoku-Cloud
# When Tenkan and Kijun are decidedly above the Cloud, the issue's trend is positive.
# When Tenkan and Kijun are decidedly below the Cloud, the issue's trend is negative.
# Using the Cloud to determine trend:
###     When prices are above the cloud, the trend is up. When prices are below the cloud, the trend is down.
###     When SenKou A is rising and above SenKouB, the uptrend is strengthening. When SenkouA is falling and below SenKouB, the downtrend is strengthening.
#### ----------
# A buy signal is reinforced when the Tenkan Sen crosses above the Kijun Sen while the Tenkan Sen, Kijun Sen, and price are all above the cloud.
# A sell signal is reinforced when the TenKan Sen crosses below the Kijun Sen while the Tenkan Sen, Kijun Sen, and price are all below the cloud.
#
#### ----------
# There are five plots that make up the Ichimoku Cloud indicator. Their names and calculations are:
# TenkanSen (Conversion Line): (High + Low) / 2 default period = 9
# KijunSen (Base Line): (High + Low) / 2 default period = 26 
# Chiku Span (Lagging Span): Price Close shifted back 26 bars 
# Senkou A (Leading Span A): (TenkanSen + KijunSen) / 2 (Senkou A is shifted forward 26 bars) 
# Senkou B (Leading Span B): (High + Low) / 2 using period = 52 (Senkou B is shifted forward 26 bars)
#### ----------

# ── User settings ─────────────────────────────────────────────────────────────
# SYMBOL       = "EURUSD=X"   # e.g. "EURUSD=X", "USDJPY=X", "XAUUSD=X", "BTC-USD"
# START        = "2024-01-01" # pull ~2 years; adjust as needed
# INTERVAL     = "4h"         # 4-hour candles
# CASH         = 10_000
# COMMISSION   = 0.000       # 0.02%

# Ichimoku params (defaults)
TENKAN       = 9
KIJUN        = 26
SENKOU_B     = 52

# Risk settings (ATR-based)
ATR_LEN      = 14
ATR_MULT_SL  = 2.0          # SL = ATR * this
ATR_MULT_TP  = 4.0          # TP = ATR * this  (≈ 2R by default)

SYMBOL       = "AUDUSD=X" #AUDUSD=X" #"USDCHF=X"  GBPUSD=X  # e.g. "EURUSD=X", "USDJPY=X", "XAUUSD=X", "BTC-USD", GBPJPY=X 
START        = "2023-10-01" # pull ~1-2 years; adjust as needed
END         = "2024-10-01" 
INTERVAL     = "4h"         # 4-hour candles
CASH         = 1000000
COMMISSION   = 0.0002      # 0.02%
df = fetch_data(symbol=SYMBOL, start=START, end=END, interval=INTERVAL)
df = add_ichimoku(df, TENKAN, KIJUN, SENKOU_B)
df["EMA"] = ta.ema(close=df["Close"], length=100)




# Fetch data from Yahoo Finance
def fetch_data(symbol: str, start: str, end:str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False, threads=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} @ {interval}. "
                         "Try a different symbol/interval or earlier START.")

    # Handle new yfinance MultiIndex format (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Extract the "Price" level for the requested symbol
        try:
            df = df.xs(symbol, axis=1, level=1)  # Keep only this ticker’s data
        except KeyError:
            # Some yfinance versions put symbol uppercase/lowercase differently
            possible = [lev for lev in df.columns.levels[1]]
            raise KeyError(f"Symbol '{symbol}' not found in MultiIndex columns. "
                           f"Available: {possible}")
    else:
        # Older yfinance already returns flat columns
        pass

    # Ensure column names are standardized
    df.columns = [c.title() for c in df.columns]
    return df.dropna()

# Ichimoku Plot
# Span_a = Senkou A
# Span_b = Senkou B
def _ichimoku_manual(df: pd.DataFrame, tenkan: int, kijun: int, senkou_b: int) -> pd.DataFrame:
    """
    Bias-safe Ichimoku (raw values for signal logic).
    - SpanA/SpanB are UNshifted.
    - Chikou: provide plotting version ONLY; logic uses past-aligned booleans.
    """
    h, l, c = df["High"], df["Low"], df["Close"]

    tenkan_line = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2.0
    kijun_line  = (h.rolling(kijun ).max() + l.rolling(kijun ).min()) / 2.0
    span_a_raw  = (tenkan_line + kijun_line) / 2.0                  # raw (no forward shift)
    span_b_raw  = (h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2.0  # raw

    # For charts only: the classic "lagging" line plotted back kijun periods.
    # DO NOT use ich_chikou_plot in entry/exit logic.
    chikou_plot = c.shift(-kijun)

    out = df.copy()
    out["ich_tenkan"]       = tenkan_line
    out["ich_kijun"]        = kijun_line
    out["ich_spanA"]        = span_a_raw
    out["ich_spanB"]        = span_b_raw
    out["ich_chikou_plot"]  = chikou_plot

    # Bias-free chikou confirmations (optional for logic):
    cloud_top = out[["ich_spanA", "ich_spanB"]].max(axis=1)
    cloud_bot = out[["ich_spanA", "ich_spanB"]].min(axis=1)

    # At time t, check what was true 26 bars ago: close[t-26] vs cloud[t-26]
    out["chik_ok_long"]  = c.shift(kijun) > cloud_top.shift(kijun)
    out["chik_ok_short"] = c.shift(kijun) < cloud_bot.shift(kijun)

    return out



def add_ichimoku(df: pd.DataFrame,
                 tenkan: int = TENKAN,
                 kijun: int = KIJUN,
                 senkou_b: int = SENKOU_B) -> pd.DataFrame:
    """
    Build bias-safe Ichimoku columns for SIGNAL logic.
    - Prefer pandas_ta for Tenkan/Kijun if available, but compute SpanA/SpanB ourselves (raw).
    - Never use a forward-shifted SpanA/SpanB.
    - Provide chikou *plotting* series and bias-free chikou booleans for logic.
    """
    out = df.copy()

    # Try to get Tenkan & Kijun from pandas_ta (core frame only), but do NOT trust spans blindly.
    tenkan_series, kijun_series = None, None
    try:
        res = ta.ichimoku(
            high=out["High"], low=out["Low"], close=out["Close"],
            tenkan=tenkan, kijun=kijun, senkou=senkou_b
        )
        ichi_core = res[0] if isinstance(res, tuple) else (res if isinstance(res, pd.DataFrame) else None)

        if isinstance(ichi_core, pd.DataFrame) and not ichi_core.empty:
            # Be explicit: pick exact ITS_/IKS_ columns for our periods only.
            its_col = f"ITS_{tenkan}"
            iks_col = f"IKS_{kijun}"
            if its_col in ichi_core.columns and iks_col in ichi_core.columns:
                tenkan_series = ichi_core[its_col]
                kijun_series  = ichi_core[iks_col]
    except Exception:
        pass  # fall back to manual fully

    # If ta not available or columns missing, compute manually.
    if tenkan_series is None or kijun_series is None:
        h, l = out["High"], out["Low"]
        tenkan_series = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2.0
        kijun_series  = (h.rolling(kijun ).max() + l.rolling(kijun ).min()) / 2.0

    # Compute raw spans (no forward shift)
    h, l, c = out["High"], out["Low"], out["Close"]
    span_a_raw = (tenkan_series + kijun_series) / 2.0
    span_b_raw = (h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2.0

    out["ich_tenkan"] = tenkan_series
    out["ich_kijun"]  = kijun_series
    out["ich_spanA"]  = span_a_raw
    out["ich_spanB"]  = span_b_raw

    # Plotting-only lagging line:
    out["ich_chikou_plot"] = c.shift(-kijun)

    # Bias-free chikou confirmations for logic:
    cloud_top = out[["ich_spanA", "ich_spanB"]].max(axis=1)
    cloud_bot = out[["ich_spanA", "ich_spanB"]].min(axis=1)
    out["chik_ok_long"]  = c.shift(kijun) > cloud_top.shift(kijun)
    out["chik_ok_short"] = c.shift(kijun) < cloud_bot.shift(kijun)

    # ATR
    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=ATR_LEN)

    # Drop warmup NaNs (needs max of 52 and ATR_LEN history)
    cols_needed = ["ich_tenkan","ich_kijun","ich_spanA","ich_spanB","ATR","chik_ok_long","chik_ok_short"]
    out = out.dropna(subset=cols_needed)
    return out

def MovingAverageSignal(df: pd.DataFrame, back_candles: int = 5) -> pd.DataFrame:
    """
    Add a single-column EMA trend signal to the DataFrame.

    Rules (evaluated per bar, using *only* current/past data):
      +1 (uptrend):   For the window [t-back_candles .. t], EVERY bar has
                      Open > EMA and Close > EMA.
      -1 (downtrend): For the same window, EVERY bar has
                      Open < EMA and Close < EMA.
       0 otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'Open', 'Close', 'EMA'.
    back_candles : int
        Number of *previous* candles to include in addition to the current one.
        Effective window size = back_candles + 1.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new integer column 'EMA_signal' in {-1, 0, +1}.
    """
    out = df.copy()

    required = ["Open", "Close", "EMA"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Window size: current bar + `back_candles` bars behind it
    w = int(back_candles) + 1
    if w <= 0:
        raise ValueError("back_candles must be >= 0")

    # Booleans per-bar relative to EMA
    above = (out["Open"] > out["EMA"]) & (out["Close"] > out["EMA"])
    below = (out["Open"] < out["EMA"]) & (out["Close"] < out["EMA"])

    # "All true in the last w bars" via rolling sum == w
    above_all = (above.rolling(w, min_periods=w).sum() == w)
    below_all = (below.rolling(w, min_periods=w).sum() == w)

    # Single signal column
    signal = np.where(above_all, 1, np.where(below_all, -1, 0)).astype(int)
    out["EMA_signal"] = signal

    return out

df = MovingAverageSignal(df, back_candles=10)


import plotly.graph_objects as go


def plot_signals_ichimoku(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    show_cloud: bool = True,
    title: str | None = None,
    offset_frac: float = 0.006,
    marker_size: int = 12,
    fig_width: int = 1000,
    fig_height: int = 700,
    show: bool = True,
):
    """
    Plot a candlestick slice with optional Ichimoku cloud, EMA, and signal markers.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe (indexed by datetime or integer).
    start_idx, end_idx : int
        Inclusive slice bounds on row positions (iloc-based).
    show_cloud : bool
        If True, overlays ich_spanA/B cloud.
    title : str | None
        Optional plot title.
    offset_frac : float
        Fraction of price used to nudge triangle markers away from candle extremes.
    marker_size : int
        Size of signal triangle markers.
    fig_width, fig_height : int
        Dimensions of the Plotly figure (in pixels).
    show : bool
        If True, immediately render the figure; otherwise just return it.
    """

    # Slice
    data = df.iloc[start_idx:end_idx + 1].copy()
    if data.empty:
        raise ValueError("Selected slice is empty. Check start_idx/end_idx.")

    for col in ["Open","High","Low","Close","signal"]:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")

    x = data.index
    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=x,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ))

    # Ichimoku cloud
    if show_cloud:
        for col in ["ich_spanA","ich_spanB"]:
            if col not in data.columns:
                raise KeyError(f"show_cloud=True but missing column: {col}")
        spanA, spanB = data["ich_spanA"], data["ich_spanB"]
        fig.add_trace(go.Scatter(x=x, y=spanA, mode="lines", name="Span A", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=x, y=spanB, mode="lines", name="Span B",
                                 fill="tonexty", opacity=0.2, line=dict(width=1)))

    # EMA
    if "EMA" in data.columns:
        fig.add_trace(go.Scatter(
            x=x, y=data["EMA"], mode="lines", name="EMA",
            line=dict(color="blue", width=2, dash="dot")
        ))

    # Offset for markers
    pad = offset_frac * data["Close"].abs().replace(0, np.nan).fillna(method="ffill").fillna(method="bfill")

    # Long markers
    bull = data["signal"] == 1
    if bull.any():
        fig.add_trace(go.Scatter(
            x=x[bull],
            y=(data.loc[bull, "Low"] - pad.loc[bull]),
            mode="markers",
            name="Long signal",
            marker=dict(symbol="triangle-up", size=marker_size, color="green"),
            hovertemplate="Long signal<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
        ))

    # Short markers
    bear = data["signal"] == -1
    if bear.any():
        fig.add_trace(go.Scatter(
            x=x[bear],
            y=(data.loc[bear, "High"] + pad.loc[bear]),
            mode="markers",
            name="Short signal",
            marker=dict(symbol="triangle-down", size=marker_size, color="red"),
            hovertemplate="Short signal<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
        ))

    # Layout
    fig.update_layout(
        title=title or "Signals, Ichimoku & EMA",
        width=fig_width,
        height=fig_height,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0)
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    if show:
        fig.show()

    return fig



plot_signals_ichimoku(df=df, start_idx=350, end_idx=450, show=False)

from backtesting import Strategy

class SignalStrategy(Strategy):
    """Generic signal-based strategy with ATR SL and RR-based TP."""
    
    atr_mult_sl: float = 1.5   # stop-loss distance = atr * atr_mult_sl
    rr_mult_tp:  float = 2.0  # take-profit distance = SL distance * rr_mult_tp

    def init(self):
        return

    def next(self):
        i = -1
        signal = int(self.data.signal[i])   # +1 long, -1 short, 0 none
        close  = float(self.data.Close[i])
        atr    = float(self.data.ATR[i])

        if not (atr > 0):
            return

        # --- manage open trades ---
        if self.position:
            # Do nothing, let SL/TP handle exits
            return

        # --- new entry ---
        sl_dist = atr * self.atr_mult_sl
        tp_dist = sl_dist * self.rr_mult_tp

        if signal == 1:  # long entry
            sl = close - sl_dist
            tp = close + tp_dist
            self.buy(size=0.99, sl=sl, tp=tp)

        elif signal == -1:  # short entry
            sl = close + sl_dist
            tp = close - tp_dist
            self.sell(size=0.99, sl=sl, tp=tp)


def run_backtest(symbol: str,
                 start: str,
                 end: str,
                 interval: str,
                 cash: float,
                 commission: float,
                 show_plot: bool = True):

    df = fetch_data(symbol, start, end, interval)
    df = add_ichimoku(df, TENKAN, KIJUN, SENKOU_B)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LEN)
    df["EMA"] = ta.ema(df.Close, length=100)

    # Make sure your EMA_signal + Ichimoku_signal were created beforehand:
    df = MovingAverageSignal(df, back_candles=7)
    df = createSignals(df, lookback_window=10, min_confirm=7)

    df = df.dropna()

    bt = Backtest(
        df,
        SignalStrategy,
        cash=cash,
        commission=commission,
        trade_on_close=True,
        exclusive_orders=True,
        margin=1/10,
    )

    stats = bt.run()
    print(f"\n=== {symbol} — Signal Strategy ===")
    print(stats)

    if show_plot:
        bt.plot(open_browser=False)
    return stats,df,bt


# Examples:
# - FX majors:  "EURUSD=X", "USDJPY=X", "GBPUSD=X"
# - Gold spot:  "XAUUSD=X"
# - Crypto:     "BTC-USD"

# ── User settings ─────────────────────────────────────────────────────────────
SYMBOL       = "USDCHF=X" #AUDUSD=X" #"USDCHF=X"  GBPUSD=X  # e.g. "EURUSD=X", "USDJPY=X", "XAUUSD=X", "BTC-USD", GBPJPY=X 
START        = "2023-10-01" # pull ~1-2 years; adjust as needed
END         = "2024-10-01" 
INTERVAL     = "4h"         # 4-hour candles
CASH         = 1000000
COMMISSION   = 0.0002      # 0.02%

stats, df, bt = run_backtest(symbol=SYMBOL, start=START, end=END, interval=INTERVAL,
                cash=CASH, commission=COMMISSION)




# ── Basket to test ────────────────────────────────────────────────────────────
SYMBOLS = [
    # FX majors
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCHF=X", "USDCAD=X", "NZDUSD=X"
]

# Global settings (override if you wish)
START      = "2023-10-01"
END        = "2024-10-01"
INTERVAL   = "4h"
CASH       = 1_000_000
COMMISSION = 0.0002

def run_all_assets(symbols=SYMBOLS,
                   start=START, end=END, interval=INTERVAL,
                   cash=CASH, commission=COMMISSION,
                   show_plot=False):
    def sget(stats, key, default=np.nan):
        try:
            return float(stats.get(key, default))
        except Exception:
            return default

    rows = []
    for sym in symbols:
        try:
            res = run_backtest(symbol=sym, start=start, end=end, interval=interval,
                               cash=cash, commission=commission, show_plot=show_plot)
            stats = res[0] if isinstance(res, (tuple, list)) else res

            rows.append({
                "Symbol": sym,
                "Return [%]":         sget(stats, "Return [%]"),
                "MaxDD [%]":          sget(stats, "Max. Drawdown [%]"),
                "AvgDD [%]":          sget(stats, "Avg. Drawdown [%]"),
                "Win Rate [%]":       sget(stats, "Win Rate [%]"),
                "Trades":             sget(stats, "# Trades"),          # ← correct key
                "Exposure Time [%]":  sget(stats, "Exposure Time [%]"),
            })
        except Exception as e:
            print(f"⚠️ {sym}: backtest failed -> {e}")
            rows.append({
                "Symbol": sym,
                "Return [%]": np.nan, "MaxDD [%]": np.nan, "AvgDD [%]": np.nan,
                "Win Rate [%]": np.nan, "Trades": np.nan, "Exposure Time [%]": np.nan
            })

    df_summary = pd.DataFrame(rows)

    # Simple (unweighted) averages ignoring NaNs
    avg_row = {"Symbol": "AVERAGE"}
    for col in ["Return [%]", "MaxDD [%]", "AvgDD [%]", "Win Rate [%]", "Trades", "Exposure Time [%]"]:
        avg_row[col] = df_summary[col].mean(skipna=True)

    df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)

    with pd.option_context("display.float_format", "{:,.2f}".format):
        print("\n=== Multi-asset backtest summary ===")
        print(df_summary)

    return df_summary

# Run it
summary = run_all_assets()

# Build numeric grids
atr_values = list(np.arange(1.0, 2.5, 0.1))
rr_values  = list(np.arange(1.0, 3.0, 0.1))

# Backtest
bt = Backtest(
    df,
    SignalStrategy,
    cash=100000,
    commission=0.0002,
    trade_on_close=True,
    exclusive_orders=True,
    margin=1/10,
)

# Optimize. return_heatmap=True yields a DataFrame with all trials.
stats, heat = bt.optimize(
    atr_mult_sl = atr_values,
    rr_mult_tp  = rr_values,
    maximize    = "Return [%]",
    return_heatmap = True,
)
print(stats._strategy)
stats


import plotly.express as px

def plot_heatmap(
    heat,
    metric_name: str = "Return [%]",
    fig_width: int = 1000,
    fig_height: int = 700,
    cmap: str = "Viridis",
    annotate: bool = True,
    min_return: float | None = None,
    max_return: float | None = None,
):
    """
    Plot an optimization heatmap (ATR x RR) with optional threshold masking.
    
    Parameters
    ----------
    heat : pd.Series or pd.DataFrame
        If Series: MultiIndex (atr_mult_sl, rr_mult_tp) -> metric values.
        If DataFrame: columns must include ['atr_mult_sl','rr_mult_tp', metric_name].
    metric_name : str
        Metric to display (e.g., 'Return [%]').
    fig_width, fig_height : int
        Figure size in pixels.
    cmap : str
        Plotly colorscale.
    annotate : bool
        Annotate cells with metric numbers.
    min_return, max_return : float | None
        Inclusive thresholds. Cells outside [min_return, max_return] are blacked out.
        If None, the side is unbounded.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    best_info : dict or None
        {'atr': ..., 'rr': ..., 'value': ...} for the best cell within thresholds,
        or None if no cell meets the thresholds.
    """
    # --- Normalize input to a pivot (rows=atr_mult_sl, cols=rr_mult_tp) ---
    if isinstance(heat, pd.Series):
        heat_df = heat.to_frame(name=metric_name).reset_index()
    else:
        if isinstance(heat.index, pd.MultiIndex) and heat.shape[1] == 1:
            heat_df = heat.reset_index()
            heat_df.columns = ["atr_mult_sl", "rr_mult_tp", metric_name]
        elif {"atr_mult_sl", "rr_mult_tp", metric_name}.issubset(heat.columns):
            heat_df = heat[["atr_mult_sl", "rr_mult_tp", metric_name]].reset_index(drop=True)
        else:
            raise ValueError("Unrecognized 'heat' format. Provide a Series or a DataFrame with "
                             "columns ['atr_mult_sl','rr_mult_tp', metric_name].")

    zdf = (heat_df
           .pivot(index="atr_mult_sl", columns="rr_mult_tp", values=metric_name)
           .sort_index()
           .sort_index(axis=1))

    # Numeric axis labels
    x_vals = zdf.columns.to_numpy(dtype=float)  # RR multipliers
    y_vals = zdf.index.to_numpy(dtype=float)    # ATR multipliers
    Z = zdf.values.astype(float)

    # --- Build base heatmap (Plotly Express for easy colorbar/labels) ---
    fig = px.imshow(
        Z,
        x=x_vals,
        y=y_vals,
        aspect="auto",
        color_continuous_scale=cmap,
        origin="lower",
        labels=dict(x="RR multiplier (TP = SL × RR)", y="ATR multiplier (SL = ATR × m)", color=metric_name),
        title=f"Optimization heatmap — {metric_name}",
    )
    fig.update_layout(width=fig_width, height=fig_height)

    if annotate:
        fig.update_traces(
            text=np.where(np.isnan(Z), "", np.round(Z, 2).astype(str)),
            texttemplate="%{text}",
            hovertemplate="ATR=%{y}<br>RR=%{x}<br>"+metric_name+"=%{z}<extra></extra>"
        )

    # --- Threshold masking: blackout cells outside [min_return, max_return] ---
    mask = np.zeros_like(Z, dtype=float)  # 0 = keep (transparent), 1 = blackout
    if (min_return is not None) or (max_return is not None):
        lower_ok = (Z >= (min_return if min_return is not None else -np.inf))
        upper_ok = (Z <= (max_return if max_return is not None else  np.inf))
        in_range = lower_ok & upper_ok
        mask = (~in_range).astype(float)

        # Add a semi-opaque black overlay for masked cells
        fig.add_trace(go.Heatmap(
            z=mask,
            x=x_vals,
            y=y_vals,
            showscale=False,
            colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(0,0,0,0.82)"]],
            hoverinfo="skip",
        ))

        # --- Find best cell within thresholds (maximize metric) ---
        if in_range.any():
            # Get index of max within allowed region
            Z_masked = np.where(in_range, Z, -np.inf)
            best_flat = np.nanargmax(Z_masked)
            best_i, best_j = np.unravel_index(best_flat, Z_masked.shape)
            best_info = {"atr": float(y_vals[best_i]), "rr": float(x_vals[best_j]), "value": float(Z[best_i, best_j])}

            # Add annotation marker
            fig.add_trace(go.Scatter(
                x=[x_vals[best_j]], y=[y_vals[best_i]],
                mode="markers+text",
                text=[f"★ {best_info['value']:.2f}"],
                textposition="top center",
                marker=dict(size=12, color="white", line=dict(width=2, color="black")),
                name="Best in range"
            ))
        else:
            best_info = None
    else:
        # No thresholds -> best over all cells (optional)
        best_flat = np.nanargmax(Z)
        best_i, best_j = np.unravel_index(best_flat, Z.shape)
        best_info = {"atr": float(y_vals[best_i]), "rr": float(x_vals[best_j]), "value": float(Z[best_i, best_j])}
        fig.add_trace(go.Scatter(
            x=[x_vals[best_j]], y=[y_vals[best_i]],
            mode="markers+text",
            text=[f"★ {best_info['value']:.2f}"],
            textposition="top center",
            marker=dict(size=12, color="white", line=dict(width=2, color="black")),
            name="Best overall"
        ))

    # Category ticks for neat labels
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=x_vals,
        ticktext=[f"{v:.1f}" for v in x_vals],
    )
    fig.update_yaxes(
        type="category",
        tickmode="array",
        tickvals=y_vals,
        ticktext=[f"{v:.1f}" for v in y_vals],
    )

    return fig


plot_heatmap(heat, metric_name="Return [%]", min_return=1, max_return=None)


import plotly.graph_objects as go

def plot_ichimoku(
    df: pd.DataFrame,
    title: str = "Ichimoku Cloud (4H)",
    kijun_periods: int = 26,
    shift_cloud_forward: bool = True,
    show_chikou: bool = True,
    show_atr: bool = False,
    cloud_eps: float | None = None,  # tolerance to avoid rapid bull/bear flipping
):
    """
    Expects columns:
      ['Open','High','Low','Close','Volume','ich_tenkan','ich_kijun',
       'ich_spanA','ich_spanB','ich_chikou','ATR']
    Index must be a DatetimeIndex (tz-aware is fine).
    """
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Derive a sensible default tolerance if not set (tiny fraction of price)
    if cloud_eps is None:
        cloud_eps = max(1e-8, float(df["Close"].abs().median()) * 1e-6)

    # Bar spacing (assumes regular series)
    if len(df.index) >= 2:
        bar_delta = pd.Series(df.index).diff().median()
        if pd.isna(bar_delta):
            bar_delta = pd.Timedelta(hours=4)
    else:
        bar_delta = pd.Timedelta(hours=4)

    x_main = df.index
    x_cloud = x_main + kijun_periods * bar_delta if shift_cloud_forward else x_main
    x_chikou = x_main - kijun_periods * bar_delta

    spanA = df["ich_spanA"]
    spanB = df["ich_spanB"]

    # Masks with tolerance to reduce flicker
    diff = spanA - spanB
    bull_mask = diff > cloud_eps
    bear_mask = diff < -cloud_eps
    # Treat very-flat parts (|diff|<=eps) as continuation of the previous regime
    flat_mask = ~(bull_mask | bear_mask)
    regime = pd.Series(np.where(bull_mask, 1, np.where(bear_mask, -1, 0)), index=df.index)
    # forward/backward fill flats
    regime = regime.replace(0, np.nan).ffill().bfill().fillna(0).astype(int)
    bull_mask = regime == 1
    bear_mask = regime == -1

    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=x_main, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))

    # Tenkan & Kijun
    fig.add_trace(go.Scatter(
        x=x_main, y=df["ich_tenkan"], name="Tenkan", mode="lines",
        line=dict(width=1.5, color="#2962ff")
    ))
    fig.add_trace(go.Scatter(
        x=x_main, y=df["ich_kijun"], name="Kijun", mode="lines",
        line=dict(width=1.5, color="#ff6d00")
    ))

    # Helper: add filled cloud segments for contiguous True blocks
    def add_cloud_segments(mask: pd.Series, fillcolor: str, showlabel: str):
        # group contiguous blocks where mask is True
        grp_id = (mask != mask.shift()).cumsum()
        first_legend = True
        for g, sub in mask.groupby(grp_id):
            if not sub.iloc[0]:  # we only draw for True segments
                continue
            idx = sub.index
            xa = x_cloud[df.index.get_indexer_for(idx)]
            ya_top = spanA.loc[idx]
            yb_bot = spanB.loc[idx]

            # Upper line (SpanA) for this block
            fig.add_trace(go.Scatter(
                x=xa, y=ya_top, mode="lines",
                line=dict(width=1, color="rgba(33,150,243,0.7)"),
                showlegend=first_legend, name=showlabel
            ))
            # Lower line (SpanB) + fill to previous
            fig.add_trace(go.Scatter(
                x=xa, y=yb_bot, mode="lines",
                line=dict(width=1, color="rgba(244,67,54,0.7)"),
                fill="tonexty", fillcolor=fillcolor,
                showlegend=False, hoverinfo="x+y"
            ))
            first_legend = False

    # Bullish (green) and Bearish (red) cloud segments
    add_cloud_segments(bull_mask, fillcolor="rgba(0,200,0,0.18)", showlabel="Cloud (Bull)")
    add_cloud_segments(bear_mask, fillcolor="rgba(200,0,0,0.18)", showlabel="Cloud (Bear)")

    # Chikou span
    if show_chikou and "ich_chikou" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_chikou, y=df["ich_chikou"], name="Chikou",
            mode="lines", line=dict(width=1.2, color="#7b1fa2", dash="dot")
        ))

    # ATR (optional, secondary y)
    if show_atr and "ATR" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_main, y=df["ATR"], name="ATR",
            mode="lines", line=dict(width=1.2, color="#455a64"), yaxis="y2"
        ))
        fig.update_layout(yaxis2=dict(title="ATR", overlaying="y", side="right", showgrid=False))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Time", rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        width=1000,
        height=800,
    )

    return fig


fig = plot_ichimoku(
    df[250:350],
    title="EURUSD 4H — Ichimoku",
    kijun_periods=26,
    shift_cloud_forward=False,   # True for classic look
    show_chikou=True,
    show_atr=False,
    cloud_eps=None  # or a value like 1e-5
)
fig.show()

