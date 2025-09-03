import os, sys
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shell_ai.trading_assistant import TradingAssistant


def _sample_data():
    closes = pd.Series(np.linspace(100, 150, 250))
    volumes = pd.Series(np.full(250, 1_000_000))
    return pd.DataFrame({'Close': closes, 'Volume': volumes})


def test_linear_regression_prediction():
    ta = TradingAssistant()
    prices = pd.Series(np.arange(100.0))
    pred, conf = ta._linear_regression_prediction(prices, 30)
    assert isinstance(pred, float)
    assert 0 <= conf <= 1


def test_portfolio_risk_and_exit():
    ta = TradingAssistant(capital=1000)
    portfolio = ta.portfolio
    pos = portfolio.enter_position('TEST', price=100, stop_loss=90, target=120, sector='Tech')
    assert pos is not None
    assert portfolio.capital == 800
    pnl = portfolio.exit_position('TEST', price=110)
    assert pytest.approx(pnl, rel=1e-6) == 20
    assert portfolio.capital == 1020


def test_generate_signal_buy_and_sell():
    ta = TradingAssistant()
    df = ta._add_indicators(_sample_data())
    df.loc[df.index[-1], ['SMA50', 'SMA200', 'RSI', 'Close']] = [100, 90, 60, 110]
    signal = ta.generate_signal(df)
    assert signal == 'buy'
    df.loc[df.index[-1], 'RSI'] = 80
    signal = ta.generate_signal(df)
    assert signal == 'sell'
