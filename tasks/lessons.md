# Lessons Learned — RAG System

_Updated as issues arise during development_

---

## 2026-04-11 — LangChain Variant Branch

### Docker: libgl1 package rename
- **What went wrong**: `libgl1-mesa-glx` was renamed to `libgl1` in Debian Bookworm (Python 3.11 base image).
  Using the old name causes `apt-get install` to fail silently or error during `docker build`.
- **Fix**: Replace `libgl1-mesa-glx` with `libgl1` in Dockerfile.
- **Pattern**: Always verify apt package names against the actual Debian version in the base image.

### Python 3.11 → 3.14 venv rebuild required
- **What went wrong**: The original venv was created with Python 3.11. Python 3.11 was later uninstalled
  (replaced by Python 3.14 via PyManager). Running `python -m venv venv` on the existing venv only
  replaced the Python link, leaving all C extension `.pyd` files compiled for `cp311` in place.
  numpy, pydantic-core, faiss-cpu all failed with `No module named '*.cp311-win_amd64'`.
- **Fix**: `rm -rf venv && python -m venv venv` then reinstall all packages fresh.
- **Pattern**: When Python version changes on a Windows machine, always do a clean venv rebuild.
  Never just repoint an existing venv to a new Python.

### Separate requirements file for Docker
- **Decision**: Created `requirements.docker.txt` with only runtime deps (no torch/numpy/scipy/scikit-learn).
  The full `requirements.txt` is a frozen pip freeze of the dev venv — it includes heavy ML deps
  that inflate the Docker image.
- **Fix**: Docker copies `requirements.docker.txt`, not `requirements.txt`.
- **Pattern**: Keep a lean `requirements.docker.txt` in sync with production needs; `requirements.txt` is for venv reproducibility.
