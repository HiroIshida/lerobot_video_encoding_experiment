import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import shutil
import imageio.v2 as imageio
from typing import Literal

import tqdm
import torch


def single_experiment(
    fast_decode: int = 0,
    codec: Literal["libsvtav1", "h264", "hevc"] = "libsvtav1",
    gop: int = 2,
):
    cond = f"{codec}_fast_decode_{fast_decode}"
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

    for i in range(5):
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

    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    ts = time.time()
    for batch in tqdm.tqdm(dataloader_train):
        pass
    elapsed = time.time() - ts
    print(f"Time taken: {elapsed:.2f} seconds")
    return elapsed

    # vcodec: str = "libsvtav1",
    # pix_fmt: str = "yuv420p",
    # g: int | None = 2,
    # crf: int | None = 30,
    # fast_decode: int = 0,
    # log_level: int | None = av.logging.ERROR,
    # overwrite: bool = False,


if __name__ == "__main__":
    t1 = single_experiment(fast_decode=0, codec="libsvtav1")
    t2 = single_experiment(fast_decode=0, codec="h264")
    t3 = single_experiment(fast_decode=0, codec="libsvtav1", gop=4)
    t4 = single_experiment(fast_decode=1, codec="h264", gop=4)
    t5 = single_experiment(fast_decode=0, codec="libsvtav1", gop=8)
    t6 = single_experiment(fast_decode=1, codec="h264", gop=8)

    print("Experiment completed.")
    print(f"libsvtav1: {t1:.2f} seconds")
    print(f"h264: {t2:.2f} seconds")
    print(f"libsvtav1 gop=4: {t3:.2f} seconds")
    print(f"h264 fast_decode=1 gop=4: {t4:.2f} seconds")
    print(f"libsvtav1 gop=8: {t5:.2f} seconds")
    print(f"h264 fast_decode=1 gop=8: {t6:.2f} seconds")
