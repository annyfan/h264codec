# ----------------------------------------------------------------------------------------------
# Copyright (C) B - All Rights Reserved
# Unauthorized copying, use, or modification to this file via any medium is strictly prohibited.
# This file is private and confidential.
# Contact: dev@b
# ----------------------------------------------------------------------------------------------

import datetime as dt
import json
import logging
import os
import re
from typing import Any, List
from torch.utils.data.sampler import Sampler

# Define the logger for this library.
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, inspired by
    torch.utils.data.sampler.SubsetRandomSampler"""

    def __init__(self, indices: List[int]) -> None:
        """
        Class initialization.

        Args:
           indices:
               The list of indices used by the sequential sampler.
        """
        self.indices = indices

    def __iter__(self):
        """
        Iterator using internal list of indices.

        Returns:
            An iterator in the internal list.
        """
        return iter(self.indices)

    def __len__(self):
        """
        Length of indices.

        Returns:
            Returns the number of elements in the list.
        """
        return len(self.indices)
