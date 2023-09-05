# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Basic datasets implement. """

import glob
import gzip
import json
import random
from contextlib import contextmanager

from paddle.io import IterableDataset


@contextmanager
def open_file(filename):
    """Construct a file handler.

    The handler can read a normal file or a file compressed by `gzip`.
    """
    if filename.endswith(".gz"):
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)
    yield fp
    fp.close()


class FileDataset(IterableDataset):
    """Single file dataset."""

    def __init__(self, filename, process_fn=None):
        self._filename = filename
        self._process_fn = process_fn

    def __iter__(self):
        with open_file(self._filename) as fin:
            for lineno, line in enumerate(fin):
                ex = json.loads(line)
                if self._process_fn is not None:
                    ex = self._process_fn(ex, self._filename)
                # ignore invalid example
                if ex is None:
                    continue
                yield ex


class FileListDataset(IterableDataset):
    """Multiple files dataset."""

    def __init__(self, filename, file_format="filelist", process_fn=None):
        if file_format == "filelist":
            self._filenames = []
            with open(filename) as fin:
                for line in fin:
                    cols = line.strip().split("\t")
                    self._filenames.append(cols[0])
        elif file_format == "glob":
            self._filenames = sorted(glob.glob(filename))
        else:
            raise ValueError(f"Unsupported file_format: {file_format}")

        self._sub_datasets = []
        for fname in self._filenames:
            self._sub_datasets.append(FileDataset(fname, process_fn=process_fn))

    def __iter__(self):
        for ds in self._sub_datasets:
            yield from ds


class MultiSourceDatset(IterableDataset):
    """Multiple source dataset."""

    def __init__(self, task_group_filename, sub_dataset_type="file", random_seed=11, process_fn=None):
        with open(task_group_filename) as fin:
            tasks = json.load(fin)
            for task in tasks:
                if "prob" not in task:
                    task["prob"] = 1
            # filter zero probability task
            tasks = [task for task in tasks if task["prob"] > 0]
            self._task_group = tasks
        if sub_dataset_type == "file":
            for task in self._task_group:
                task["dataset"] = FileDataset(task["filepath"], process_fn=process_fn)
        else:
            raise NotImplementedError("Cannot support filelist now.")
        sum_prob = sum([task["prob"] for task in self._task_group])
        for task in self._task_group:
            task["prob"] = task["prob"] / sum_prob

        self.random_seed = random_seed

    def __iter__(self):
        rng = random.Random(self.random_seed)
        probs = [task["prob"] for task in self._task_group]
        # Initialize task iterator
        for task in self._task_group:
            task["iterator"] = iter(task["dataset"])
        while True:
            task = rng.choices(self._task_group, weights=probs)[0]
            try:
                yield from task["iterator"]
            except StopIteration:
                task["iterator"] = iter(task["dataset"])
                yield from task["iterator"]
