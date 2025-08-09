import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    with open("results.json", "r") as f:
        results = json.load(f)

    codecs = ["libsvtav1", "h264"]
    gops = [2, 4, 8, 16]

    fig = plt.figure(figsize=(6.5, 5.5))

    for codec in codecs:
        times = results[codec]["times"]
        sizes_mb = [s / (1024 * 1024) for s in results[codec]["sizes"]]
        plt.plot(times, sizes_mb, marker="o", linestyle="-", label=codec)
        for t, s, g in zip(times, sizes_mb, gops):
            plt.annotate(
                f"GOP {g}",
                (t, s),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

    plt.title("Trade-off: Dataset Size vs. Decode Time", fontsize=14)
    plt.xlabel("Decode Time (seconds)", fontsize=12)
    plt.ylabel("Dataset Size (MB)", fontsize=12)
    plt.grid(True)
    plt.legend(title="Codec")
    fig.tight_layout()
    fig.savefig("paleto-plot.png", dpi=300)
    plt.close(fig)
