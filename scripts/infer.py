import logging
import signal
import sys
import time

import cv2
import hydra
import torch
from event_camera_stream import MjpegConnector
from omegaconf import DictConfig

from experimenting.agents.margipose_estimator import predict3d
from experimenting.utils.evaluation_helpers import load_model
from experimenting.utils.skeleton_helpers import Skeleton
from experimenting.utils.visualization import plot_skeleton_3d

logging.basicConfig(level=logging.INFO)


def exit_gracefully(sc, signum, frame):
    global running

    # Disconnect
    sc.disconnect()
    running = False

    # Wait one second
    time.sleep(1)
    sys.exit(0)


class HPEAccumulation:
    """ Render accumulation image and print some statistics
    """

    def __init__(self, model):
        self.start_time = int(round(time.time() * 1000))
        self.stop_time = int(round(time.time() * 1000))
        self.accumulation_bytes = 0
        self.frames = 0
        self.model = model

    @torch.no_grad()
    def callback(self, image, min_ts, max_ts, events, amplification, length, client):
        self.stop_time = int(round(time.time() * 1000))
        self.accumulation_bytes += length

        if (self.stop_time - self.start_time) >= 1000:
            print(
                "Bitrate: {0} Kb/s, Framerate: {1} fps".format(
                    (self.accumulation_bytes * 8 / 1000), self.frames
                )
            )

            self.accumulation_bytes = 0
            self.frames = 0
            self.start_time = int(round(time.time() * 1000))
            self.min_size = 0
            self.max_size = 0

        proper_img = torch.zeros((1, 1, 260, 346))

        proper_img[0, 0, :] = torch.from_numpy(cv2.resize(image[:, :, 0], (346, 260)))

        preds = self.model(proper_img.double())
        xy = preds[0][-1]
        zy = preds[1][-1]
        xz = preds[2][-1]
        normalized_skeleton = Skeleton(predict3d(xy, zy, xz)[0])
        plot_skeleton_3d(normalized_skeleton)
        print(normalized_skeleton._get_tensor())

        # Show accumulation image GRAY
        cv2.imshow("Receiver image", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            client.disconnect()

        self.frames += 1


@hydra.main(config_path="confs/train/", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    model = load_model(cfg, **cfg).double()

    acc = HPEAccumulation(model)
    sc = MjpegConnector("http://10.245.83.34:5005", acc.callback)

    if not sc.connect():
        sys.exit(0)

    # Intercept signals
    signal.signal(
        signal.SIGINT, lambda signum, frame: exit_gracefully(sc, signum, frame)
    )
    signal.signal(
        signal.SIGTERM, lambda signum, frame: exit_gracefully(sc, signum, frame)
    )

    sc.start()
    while sc.join():
        pass

    sc.disconnect()


if __name__ == "__main__":
    running = True
    main()
