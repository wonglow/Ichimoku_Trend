from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from backtesting import Backtest, Strategy
