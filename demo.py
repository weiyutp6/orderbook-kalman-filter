"""Run the filter on simulated data and report how it compares with the raw series."""
import numpy as np

from kalman_obi import simulate, run_filter


def main() -> None:
    data = simulate(n=2000, seed=1)
    est = run_filter(data)

    raw_rmse = float(np.sqrt(np.mean((data["price"] - data["true_price"]) ** 2)))
    kf_rmse = float(np.sqrt(np.mean((est[:, 0] - data["true_price"]) ** 2)))
    vel_rmse = float(np.sqrt(np.mean((est[:, 1] - data["true_vel"]) ** 2)))
    vel_naive = float(np.sqrt(np.mean(data["true_vel"] ** 2)))

    print(f"price RMSE  raw observation : {raw_rmse:.3f}")
    print(f"price RMSE  filtered        : {kf_rmse:.3f}")
    print(f"velocity RMSE filtered      : {vel_rmse:.3f}   (predicting zero: {vel_naive:.3f})")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(data["price"], alpha=0.4, label="observed")
    ax[0].plot(data["true_price"], label="true")
    ax[0].plot(est[:, 0], label="filtered")
    ax[0].legend()
    ax[0].set_title("Price")
    ax[1].plot(data["true_vel"], label="true velocity")
    ax[1].plot(est[:, 1], label="estimated velocity")
    ax[1].legend()
    ax[1].set_title("Velocity")
    plt.tight_layout()
    plt.savefig("demo.png", dpi=120)
    print("saved demo.png")


if __name__ == "__main__":
    main()
