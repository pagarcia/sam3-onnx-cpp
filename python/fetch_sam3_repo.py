#!/usr/bin/env python3
"""Clone/fetch the exact SAM3 source revision used for ONNX exports."""

from __future__ import annotations

import argparse
from pathlib import Path

from sam3_revision import PRE31_SAM3_COMMIT, SAM3_REPO_URL, ensure_sam3_revision, sync_sam3_repo


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam3-repo",
        type=Path,
        default=repo_root().parent / "sam3",
        help="Target local SAM3 checkout. Defaults to ../sam3 next to this repo.",
    )
    parser.add_argument(
        "--revision",
        default=PRE31_SAM3_COMMIT,
        help="SAM3 git revision to check out.",
    )
    parser.add_argument(
        "--repo-url",
        default=SAM3_REPO_URL,
        help="SAM3 git repository URL.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify the checkout revision; do not fetch or checkout.",
    )
    args = parser.parse_args()

    sam3_repo = args.sam3_repo.expanduser().resolve()
    if args.check_only:
        revision = ensure_sam3_revision(sam3_repo, expected=args.revision)
    else:
        revision = sync_sam3_repo(
            sam3_repo,
            expected=args.revision,
            repo_url=args.repo_url,
        )

    print(f"[OK] SAM3 repo   : {sam3_repo}")
    print(f"[OK] SAM3 commit : {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
