import numpy as np

from kalman_obi import (PriceVelocityFilter, order_book_imbalance,
                        rolling_volatility, run_filter, simulate)


def test_order_book_imbalance_bounds():
    bids = np.array([[100.0, 5.0], [99.0, 5.0]])
    asks = np.array([[101.0, 5.0], [102.0, 5.0]])
    assert abs(order_book_imbalance(bids, asks)) < 1e-6
    assert order_book_imbalance(bids, asks * [1, 0.0]) > 0.99
    assert -1.0 <= order_book_imbalance(asks, bids) <= 1.0


def test_rolling_volatility_flat_series_is_zero():
    assert rolling_volatility([10.0] * 20) == 0.0
    assert rolling_volatility([1.0]) == 0.0


def test_tracks_constant_velocity():
    rng = np.random.default_rng(3)
    t = np.arange(300)
    prices = 50 + 0.4 * t + rng.normal(0, 0.5, t.size)
    f = PriceVelocityFilter(c_obi=0.0, c_vol=0.0, velocity_decay=1.0, q=1e-4, r=0.25)
    f.initialise(prices[:10])
    for p in prices:
        f.step(p, 0.0, 0.0)
    assert abs(f.x[1] - 0.4) < 0.05


def test_filter_beats_raw_observation_on_price():
    data = simulate(n=2000, seed=1)
    est = run_filter(data)
    raw = np.sqrt(np.mean((data["price"] - data["true_price"]) ** 2))
    filt = np.sqrt(np.mean((est[:, 0] - data["true_price"]) ** 2))
    assert filt < raw


def test_covariance_stays_positive_definite():
    f = PriceVelocityFilter()
    f.initialise([100.0, 100.1, 100.2])
    for p in np.linspace(100, 101, 200):
        f.step(p, 0.1, 0.01)
    assert np.all(np.linalg.eigvalsh(f.P) > 0)
