"""Hugging Face Space entry point for the FactoryLens demo.

scripts/build_space.py puts the fitted models in models/ and the prepared
test and validation parts in data/ next to this file. The raw dataset is not
needed.
"""

import torch

from factorylens.demo import build_app

torch.set_num_threads(2)  # free Spaces have two CPU cores

app = build_app("models", root="data/raw", cache_dir="data", max_parts=60)
# one request at a time: the inspector keeps the reliability window as state
app.queue(default_concurrency_limit=1).launch()
