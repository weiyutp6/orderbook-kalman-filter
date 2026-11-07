# Order-Book Kalman Filter

A state-space filter that tracks a price and its short-term velocity, using order-book imbalance (OBI) and recent volatility as control inputs.

## Model

State `x = [price, velocity]`, observed through the price only:

    x_{k+1} = A x_k + B u_k + w_k,    w ~ N(0, Q)
    z_k     = H x_k + v_k,            v ~ N(0, R)

    A = [[1, dt], [0, decay]]          near-constant velocity (decay < 1 mean-reverts it)
    u = [OBI, volatility]
    B = [[0, 0], [c_obi*dt, c_vol*dt]] imbalance pushes velocity, volatility damps it
    H = [1, 0]

    OBI = (bid volume - ask volume) / (bid volume + ask volume)    in [-1, 1]

The dynamics are linear, so the extended Kalman filter reduces to the standard Kalman filter. The class uses explicit matrices so a nonlinear `f(x, u)` and its Jacobian can replace `A x + B u`.

`initialise()` seeds the state from a burn-in window: the last price, a fitted slope as velocity, and a diagonal covariance from the sample variances.

## Result on simulated data

The simulator draws a path from the same model (known truth), then adds observation noise. Scored over 2,000 steps:

| Quantity | Error (RMSE) |
|---|---|
| Price, raw observation | 0.982 |
| Price, filtered | 0.526 |
| Velocity, filtered | 0.181 |
| Velocity, predicting zero | 4.823 |

This shows the filter works when its model matches the data. It says nothing about real markets, where the true dynamics are unknown. Treat the constants as illustrative, not tuned.

## Files

| File | Role |
|---|---|
| `kalman_obi.py` | OBI and volatility helpers, `PriceVelocityFilter`, simulator. |
| `demo.py` | Runs the filter on simulated data, prints the table above, writes `demo.png`. |
| `test_kalman.py` | Imbalance bounds, constant-velocity tracking, filter beats raw price, covariance stays positive definite. |

## Usage

```bash
pip install -r requirements.txt
python -m pytest
python demo.py
```

```python
from kalman_obi import PriceVelocityFilter, order_book_imbalance, rolling_volatility

f = PriceVelocityFilter(dt=1.0, c_obi=0.5, c_vol=-0.2)
f.initialise(first_ten_prices)
state = f.step(price, order_book_imbalance(bids, asks), rolling_volatility(recent_prices))
```

## Disclaimer

For research and education only. Not investment advice.
