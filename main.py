import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import shutil
import imageio.v2 as imageio
from typing import Literal
import tqdm
import torch
import json


def single_experiment(
    fast_decode: int = 0,
    codec: Literal["libsvtav1", "h264", "hevc"] = "libsvtav1",
    gop: int = 2,
):
    cond = f"{codec}_gop_{gop}_fast_decode_{fast_decode}"
    dataset_path = Path(f"/tmp/{cond}")
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    features = {
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (3, 112, 112),
            "names": ["rgb", "height", "width"],
        }
    }
    dataset = LeRobotDataset.create("test", 30, features, dataset_path)

    for i in range(40):
        gif_path = Path("./gifs") / f"{i}.gif"
        rbgs = [frame[:, :, :3] for frame in imageio.mimread(gif_path)]
        for rgb in rbgs:
            frame = {"observation.images.rgb": rgb.transpose(2, 0, 1)}
            dataset.add_frame(frame, "test")
        opt = {
            "vcodec": codec,
            "fast_decode": fast_decode,
            "g": gop,
        }
        dataset.save_episode(video_encode_option=opt)

    total_size = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file())

    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    ts = time.time()
    for _ in range(5):
        for batch in tqdm.tqdm(dataloader_train, desc=f"Decoding {codec} gop={gop}"):
            pass
    elapsed = (time.time() - ts) / 5  # Average over 5 iterations

    print(f"Codec: {codec}, GOP: {gop}")
    print(f"  - Dataset Size: {total_size / (1024 * 1024):.2f} MB")
    print(f"  - Decode Time: {elapsed:.2f} seconds")

    shutil.rmtree(dataset_path)
    return elapsed, total_size


if __name__ == "__main__":
    codecs = ["libsvtav1", "h264"]

    gops = [2, 4, 8, 16]
    fast_decode = 0

    results = {codec: {"times": [], "sizes": []} for codec in codecs}

    for codec in codecs:
        for gop in gops:
            elapsed_time, dataset_size = single_experiment(
                fast_decode=fast_decode,
                codec=codec,
                gop=gop,
            )
            results[codec]["times"].append(elapsed_time)
            results[codec]["sizes"].append(dataset_size)

    print("\nAll experiments completed.")
    # save as json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results.json")
