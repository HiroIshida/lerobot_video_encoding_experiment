import subprocess
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import shutil

if __name__ == "__main__":   
    features = {
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (3, 112, 112),
            "names": ["rgb", "height", "width"],
        }
    }
    shutil.rmtree("/tmp/_dataset", ignore_errors=True)
    dataset = LeRobotDataset.create("test", 30, features, "/tmp/_dataset")
    dummy_frame = np.random.randint(0, 255, (3, 112, 112), dtype=np.uint8)
    for _ in range(1000):
        dataset.add_frame(frame = {"observation.images.rgb": dummy_frame}, task="test")
    dataset.save_episode()  # no error
