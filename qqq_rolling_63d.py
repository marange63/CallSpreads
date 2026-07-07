import yfinance as yf
import matplotlib.pyplot as plt

WINDOW = 63

df = yf.download("QQQ", period="3y", auto_adjust=True, progress=False)
close = df["Close"].squeeze()

rolling_return = (close / close.shift(WINDOW) - 1) * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(rolling_return.index, rolling_return.values, linewidth=1.0, color="#1f77b4")
ax.axhline(0, color="black", linewidth=0.5)
ax.fill_between(rolling_return.index, rolling_return.values, 0,
                where=(rolling_return.values >= 0), alpha=0.15, color="green")
ax.fill_between(rolling_return.index, rolling_return.values, 0,
                where=(rolling_return.values < 0), alpha=0.15, color="red")
ax.set_title(f"QQQ rolling {WINDOW} trading-day cumulative return (~3 months)")
ax.set_ylabel("Return (%)")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)

latest = rolling_return.dropna().iloc[-1]
ax.annotate(f"latest: {latest:+.2f}%",
            xy=(rolling_return.dropna().index[-1], latest),
            xytext=(10, 10), textcoords="offset points",
            fontsize=10, fontweight="bold")

plt.tight_layout()
out = "qqq_rolling_63d.png"
plt.savefig(out, dpi=120)
print(f"saved {out}")
print(f"latest {WINDOW}d return: {latest:+.2f}%")
print(f"min: {rolling_return.min():+.2f}%  max: {rolling_return.max():+.2f}%  mean: {rolling_return.mean():+.2f}%")
