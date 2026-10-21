# Checkpoint Storage Strategy

## Recommendation

Do not keep model weights hard-coded inside application source files.
Instead:

- define model names in `configs/model_sources.yaml`,
- document download locations in `models/README.md`,
- resolve paths through `src/waste_sorting_vision/config.py`.

For this repository, keeping the `waste_v*.pt` checkpoints under `models/` is acceptable because it allows the demo app to run immediately after setup.
If repository size becomes a concern, the same configuration can later be redirected to GitHub Releases, Git LFS, or another artefact store.

## Practical Options

### Option A — GitHub Releases
Good when you want a clean code repo and downloadable fixed artefacts.

### Option B — Git LFS
Good when you want weights versioned with the repository.

### Option C — External Model Repository
Good when you want a dedicated place for checkpoints, metadata, and future variants.

## Recommended Default for This Project

For this project, immediate demo readiness is a reasonable default.
If the repository later needs to be slimmed down, move the checkpoints out of Git and keep the configuration interface unchanged.
