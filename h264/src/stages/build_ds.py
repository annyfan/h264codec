# ----------------------------------------------------------------------------------------------
# Copyright (C) Botni.Vision, Inc - Montreal, QC, Canada - All Rights Reserved
# Unauthorized copying, use, or modification to this file via any medium is strictly prohibited.
# This file is private and confidential.
# Contact: dev@botni.vision
# ----------------------------------------------------------------------------------------------
import argparse
import csv
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import random

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
        default=0.15,
        help="The ratio [0-1] for the validation set.",
    )

    args = parser.parse_args()

    return args


def build_dataset(args: argparse.Namespace) -> None:
    """_summary_

    Args:
        args (argparse.Namespace): _description_
    """
    # Set the logger configuration.
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.getLevelName("INFO"),
        stream=sys.stdout,
    )

    # Load the JSON source file.
    df = pd.read_json(Path(args.base_path) / "dataset.json")

    validationSetSize = int(args.val_ratio * df.shape[0])
    validationSet = random.choices(df["h264"].tolist(), k=validationSetSize)

    header = ["id", "flag"]
    with open(Path(args.base_path) / "dataset.csv", "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for index, row in df.iterrows():
            trainFlag = "VALIDATE" if row["h264"] in validationSet else "TRAIN"
            writer.writerow([row["h264"].replace(".h264", ""), trainFlag])

    LOG.info("Done.")


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args)
