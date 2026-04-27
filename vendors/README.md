# Vendors

This directory is for local reference checkouts of established atomistic and electronic-structure codes. The directory name is conventional; these trees are not vendored dependencies.

Reference trees are intentionally not part of the Python package, not imported by `mlx_atomistic`, and not built by `uv sync`. They are used for architecture study, algorithm references, and validation planning.

Tracked files:

- `vendor-lock.json`: records reference checkout URLs, branches, shallow-clone intent, and observed commits.

Ignored files:

- All reference source trees under this directory.

Fetch/update reference checkouts with:

```bash
scripts/fetch-vendors
```
