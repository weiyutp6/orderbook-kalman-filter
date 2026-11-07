"""
State-space filter for price and short-term velocity, driven by order-book
imbalance (OBI) and recent volatility.

State      x = [price, velocity]
Dynamics   x_{k+1} = A x_k + B u_k + w_k,   w ~ N(0, Q)
Observed   z_k     = H x_k + v_k,           v ~ N(0, R)

with
    A = [[1, dt], [0, decay]]                near-constant velocity (decay <= 1 mean-reverts it)
    u = [obi, vol]                           control inputs
    B = [[0, 0], [c_obi*dt, c_vol*dt]]       OBI pushes velocity, volatility damps it
    H = [1, 0]                               only price is observed

The model is linear, so the extended Kalman filter reduces to the standard
Kalman filter. The class is written with explicit matrices so a nonlinear
f(x, u) and its Jacobian can be swapped in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def order_book_imbalance(bids: np.ndarray, asks: np.ndarray) -> float:
    """(bid volume - ask volume) / (bid volume + ask volume), in [-1, 1].
    bids and asks are arrays of [price, size] rows."""
    bid_vol = float(np.sum(bids[:, 1]))
    ask_vol = float(np.sum(asks[:, 1]))
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)


def rolling_volatility(prices: list[float] | np.ndarray) -> float:
    """Standard deviation of simple returns over the supplied window."""
    arr = np.asarray(prices, dtype=float)
    if arr.size < 2:
        return 0.0
    rets = np.diff(arr) / arr[:-1]
    return float(np.std(rets))


@dataclass
class PriceVelocityFilter:
    dt: float = 1.0
    c_obi: float = 0.5
    c_vol: float = -0.2
    velocity_decay: float = 0.9   # 1.0 = pure constant velocity
    q: float = 0.01          # process-noise scale
    r: float = 1.0           # measurement-noise variance
    x: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P: np.ndarray = field(default_factory=lambda: np.eye(2))

    def __post_init__(self) -> None:
        self.A = np.array([[1.0, self.dt], [0.0, self.velocity_decay]])
        self.B = np.array([[0.0, 0.0], [self.c_obi * self.dt, self.c_vol * self.dt]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * self.q
        self.R = np.array([[self.r]])

    def initialise(self, prices: list[float] | np.ndarray, obi: list[float] | None = None,
                   vol: list[float] | None = None) -> None:
        """Seed the state from a short burn-in window: last price, fitted slope,
        and a diagonal covariance from the sample variances."""
        prices = np.asarray(prices, dtype=float)
        t = np.arange(len(prices)) * self.dt
        slope = float(np.polyfit(t, prices, 1)[0]) if len(prices) >= 2 else 0.0
        self.x = np.array([prices[-1], slope])
        var_p = float(np.var(prices)) + 1e-6
        var_u = float(np.var(obi)) + float(np.var(vol)) if obi is not None else 1.0
        self.P = np.diag([var_p, var_u + 1e-6]) * 10.0

    def step(self, price: float, obi: float, vol: float) -> np.ndarray:
        """One predict-update cycle. Returns the filtered state [price, velocity]."""
        u = np.array([obi, vol])
        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        innovation = price - float((self.H @ x_pred)[0])
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + (K.flatten() * innovation)
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        return self.x.copy()


def simulate(n: int = 500, dt: float = 1.0, seed: int = 0,
             c_obi: float = 0.5, c_vol: float = -0.2, velocity_decay: float = 0.9,
             process_std: float = 0.1, obs_std: float = 1.0):
    """Generate a price path from the model itself so the filter can be scored
    against known truth. Returns dict of arrays: price, true_price, true_vel, obi, vol."""
    rng = np.random.default_rng(seed)
    obi = np.clip(np.cumsum(rng.normal(0, 0.15, n)) * 0.2 + rng.normal(0, 0.2, n), -1, 1)
    vol = np.abs(rng.normal(0.002, 0.001, n))

    true = np.zeros((n, 2))
    true[0] = [100.0, 0.0]
    for k in range(n - 1):
        drift = np.array([true[k, 1] * dt, c_obi * dt * obi[k] + c_vol * dt * vol[k] * 100])
        true[k + 1, 0] = true[k, 0] + drift[0] + rng.normal(0, process_std)
        true[k + 1, 1] = true[k, 1] * velocity_decay + drift[1] + rng.normal(0, process_std)
    price = true[:, 0] + rng.normal(0, obs_std, n)
    return {"price": price, "true_price": true[:, 0], "true_vel": true[:, 1],
            "obi": obi, "vol": vol * 100}


def run_filter(data: dict, burn_in: int = 10, **kwargs) -> np.ndarray:
    """Run the filter over a simulated or recorded series, returning [n, 2] estimates."""
    f = PriceVelocityFilter(**kwargs)
    f.initialise(data["price"][:burn_in], data["obi"][:burn_in], data["vol"][:burn_in])
    est = [f.step(p, o, v) for p, o, v in zip(data["price"], data["obi"], data["vol"])]
    return np.array(est)
