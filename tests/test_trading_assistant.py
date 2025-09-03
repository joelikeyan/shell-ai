import logging
import sys
from unittest.mock import patch

import pandas as pd

# Ensure the package root is on the path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shell_ai import trading_assistant


def test_evaluate_updates_portfolio_and_logs(caplog):
    data = pd.DataFrame({"Close": [90, 110]})

    with patch("shell_ai.trading_assistant.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = data

        portfolio = {"cash": 1000.0, "shares": 0}

        with caplog.at_level(logging.INFO):
            trading_assistant.evaluate("AAPL", portfolio)

    assert portfolio == {"cash": 1020.0, "shares": 0}

    messages = caplog.messages
    assert "Bought 1 share of AAPL at 90" in messages[0]
    assert "Sold 1 share of AAPL at 110" in messages[1]
