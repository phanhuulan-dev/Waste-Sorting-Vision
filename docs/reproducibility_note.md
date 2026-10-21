# Project Scope Note

This repository is organised as a clean standalone release of the project.
It focuses on the parts that are most useful to readers and reviewers:

- the Streamlit inference app
- the checkpoints currently exposed by the app
- demo images and videos
- class labels used in the interface
- the results report and short notes on checkpoint handling

The documentation is intentionally result-oriented.
It summarises the app behaviour and the outputs that matter most for understanding the project.

## What Is Reproducible Here

- running the Streamlit app with the current project structure
- loading the configured checkpoints through `configs/model_sources.yaml`
- reproducing the current interface behaviour for image and video inference
- verifying the small test suite for configuration and label handling
- recreating the tested local dependency set through `requirements/full-lock.txt`

## What Is Outside The Scope

- full end-to-end retraining from the original project stage
- exact recreation of the original training environment
- exact recreation of every training run, dataset state, and experiment log

This repository should therefore be read as project documentation and application code, not as a step-by-step retraining guide.
