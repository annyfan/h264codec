# ----------------------------------------------------------------------------------------------
# Copyright (C) B All Rights Reserved
# Unauthorized copying, use, or modification to this file via any medium is strictly prohibited.
# This file is private and confidential.
# Contact: dev@b
# ----------------------------------------------------------------------------------------------
import argparse
import csv
import logging
import os
import sys
from pathlib import Path
import random
from skimage import io, color
import matplotlib.pyplot as plt

LOG = logging.getLogger(os.path.basename(__file__))


def parse_args() -> argparse.Namespace:
    """
    Parse arguments for the application.

    Returns
        The data structure contaning the parameters set by the command line arguments.
    """
    format_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=format_class, description="Dataset building utility."
    )
    parser.add_argument("--debug", action="store_true", help="debug flag.")
    parser.add_argument(
        "--base-path",
        type=str,
        default="/data/bytes_learning/h264_v20230930",
        help="Dataset base folder.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="The ratio [0-1] for the validation set.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=10,
        help="The number of histogram bins.",
    )

    args = parser.parse_args()

    return args


def build_dataset(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """
    basePath = Path(args.base_path)

    header = ["id", "flag"]
    header.extend([f"h{count}" for count in range(args.hist_bins)])
    with open(basePath / "img_ds.csv", "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for image_file in sorted((basePath / "image").rglob("*.jpeg")):
            im = io.imread(image_file)
            grayIm = color.rgb2gray(im)
            histCount, _, _ = plt.hist(grayIm.flatten(), bins=args.hist_bins)

            trainFlag = (
                "VALIDATE" if random.uniform(0, 1) <= args.val_ratio else "TRAIN"
            )
            out = [image_file.stem, trainFlag]
            out.extend(histCount / sum(histCount))
            writer.writerow(out)

    LOG.info("Done.")


if __name__ == "__main__":
    args = parse_args()

    # Set the logger configuration.
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.getLevelName("DEBUG" if args.debug else "INFO"),
        stream=sys.stdout,
    )

    build_dataset(args)
