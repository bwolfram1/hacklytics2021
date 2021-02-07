# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 00:45:21 2021

@author: Brandon
"""

import pandas as pd
import ray
import numpy as np

from ray import tune
from ray.tune.registry import register_env

import tensortrade.env.default as default

from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger


cdd = CryptoDataDownload()

dataf = cdd.fetch("gemini", "USD", "BTC", "1h")
data = dataf.iloc[45000:]

def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


features = []
for c in data.columns[1:]:
    s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
    features += [s]

cp = Stream.select(features, lambda s: s.name == "close")

features = [
    cp.diff().rename("diff"),
    rsi(cp, period=14).rename("rsi"),
    macd(cp, fast=10, slow=50, signal=5).rename("macd"),
    cp.rolling(window=10).mean().rename("10ma"),
    cp.rolling(window=50).mean().rename("50ma")
]

feed = DataFeed(features)
feed.compile()

gemini = Exchange("gemini", service=execute_order)(
    Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
)

portfolio = Portfolio(USD, [
    Wallet(gemini, 1700 * USD),
    Wallet(gemini, 10 * BTC)
])

renderer_feed = DataFeed([
    Stream.source(list(data["date"])).rename("date"),
    Stream.source(list(data["open"]), dtype="float").rename("open"),
    Stream.source(list(data["high"]), dtype="float").rename("high"),
    Stream.source(list(data["low"]), dtype="float").rename("low"),
    Stream.source(list(data["close"]), dtype="float").rename("close"),
    Stream.source(list(data["volume"]), dtype="float").rename("volume")
])
chart_renderer = PlotlyTradingChart(
    display=True,  # show the chart on screen (default)
    height=800,  # affects both displayed and saved file height. None for 100% height.
    save_format="html",  # save the chart to an HTML file
    auto_open_html=False,  # open the saved HTML chart in a new browser tab
)

file_logger = FileLogger(
    filename="t1.log",  # omit or None for automatic file name
    path="training_logs"  # create a new directory if doesn't exist, None for no directory
)
env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer_feed=renderer_feed,
    renderer=chart_renderer,
    window_size=48
)



agent = DQNAgent(env)

agent.train(n_steps=5000, n_episodes=2, save_path="agents/")

