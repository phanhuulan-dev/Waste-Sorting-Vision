# Model Artifacts

Do not hard-code checkpoint paths in application source files.

Current repository setup:

- `waste_v5` (default) resolves to `models/waste_v5.pt`
- `waste_v3` resolves to `models/waste_v3.pt`
- `waste_v2` resolves to `models/waste_v2.pt`
- `best` resolves to `models/best.pt`
- `best5` resolves to `models/best5.pt`
- environment overrides:
  - `WSV_MODEL_WASTE_V5`
  - `WSV_MODEL_WASTE_V3`
  - `WSV_MODEL_WASTE_V2`
  - `WSV_MODEL_BEST`
  - `WSV_MODEL_BEST5`

The checkpoints are kept in this repository so the demo can run immediately after setup.
Configuration-based path resolution is still used so the files can be moved later if needed.

Alternative distribution strategies:

1. download from GitHub Release assets,
2. track with Git LFS,
3. host them in a separate model repository.

If repository size becomes a concern later, the same configuration can be reused with GitHub Releases, Git LFS, or another external storage location.
