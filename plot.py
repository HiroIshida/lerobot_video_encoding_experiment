import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    with open("results.json", "r") as f:
        results = json.load(f)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    codecs = ["libsvtav1", "h264"]
    gops = [2, 4, 8, 16]
    for codec in codecs:
        ax1.plot(gops, results[codec]["times"], marker="o", linestyle="-", label=codec)
    ax1.set_title("Decode Speed vs. GOP Size", fontsize=14)
    ax1.set_xlabel("GOP Size", fontsize=12)
    ax1.set_ylabel("Decode Time (seconds)", fontsize=12)
    ax1.set_xticks(gops)
    ax1.grid(True)
    ax1.legend(title="Codec")

    for codec in codecs:
        sizes_mb = [s / (1024 * 1024) for s in results[codec]["sizes"]]
        ax2.plot(gops, sizes_mb, marker="o", linestyle="-", label=codec)
    ax2.set_title("Dataset Size vs. GOP Size", fontsize=14)
    ax2.set_xlabel("GOP Size", fontsize=12)
    ax2.set_ylabel("Dataset Size (MB)", fontsize=12)
    ax2.set_xticks(gops)
    ax2.grid(True)
    ax2.legend(title="Codec")

    fig.tight_layout()
    fig.savefig("results_side_by_side.png", dpi=200)
    plt.close(fig)
