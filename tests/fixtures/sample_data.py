import pytest
import pandas as pd

@pytest.fixture
def sample_ohlcv_data():
    data = {
        'Open': [1.0, 1.1, 1.2],
        'High': [1.2, 1.3, 1.4],
        'Low': [0.9, 1.0, 1.1],
        'Close': [1.1, 1.2, 1.3],
        'Volume': [1000, 1500, 2000]
    }
    return pd.DataFrame(data)