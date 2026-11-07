# BTC Price and Velocity Filter

`btc_EKF_velocity.ipynb` runs an extended Kalman filter on the BTC order book. The state is price and velocity; order-book imbalance and recent volatility act as control inputs. Price and imbalance come from a polling thread that reads the Binance depth endpoint into a one-slot queue.

Needs `numpy`, `pandas`, `matplotlib` and `requests`. The depth endpoint is not available from every region.

For research and education only. Not investment advice.
