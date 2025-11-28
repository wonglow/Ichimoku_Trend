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
