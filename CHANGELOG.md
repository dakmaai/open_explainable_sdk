# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-09

### Added

- Initial public release of `dakma-sdk` (import as `dakma` or `dakma_sdk`).
- `DakmaClient` decorators: `track_data`, `track_training`, `explain`, `explain_integrated_gradients`.
- Governance and evaluation documentation templates for audit/decision reports.
- Markdown and HTML audit reports with SHAP and Integrated Gradients attribution.
- Optional extras: `[ml]` (pandas, scikit-learn, XGBoost, SHAP), `[dl]` (PyTorch, Captum).
- Example scripts under `examples/` for XGBoost, SVM, and PyTorch MLP workflows.
- Unit and smoke tests (`pytest`).

### Changed

- Compliance language: documentation-aid disclaimers only; no certification checkmarks.
- Repository hygiene: `.gitignore`, `MANIFEST.in`; build artifacts no longer tracked.

### Notes

- This is an **alpha** release (`Development Status :: 3 - Alpha`).
- Reports and governance blocks are **documentation aids**, not legal or regulatory certification.

[0.1.0]: https://github.com/dakmaai/open_explainable_sdk/releases/tag/v0.1.0
