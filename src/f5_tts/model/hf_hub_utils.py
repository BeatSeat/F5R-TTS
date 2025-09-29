from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi


def _is_enabled(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value not in {"", "0", "false", "False"}


def push_checkpoint_to_hub(
    local_path: str,
    *,
    step: int,
    last: bool = False,
) -> None:
    """Upload a checkpoint file to the configured Hugging Face Hub repository.

    Uploading is skipped unless ``F5R_TTS_HF_REPO_ID`` is configured. When
    ``F5R_TTS_HF_STRICT`` is truthy, any failure during upload raises an
    exception; otherwise the error is logged and training continues.
    """

    repo_id = os.environ.get("F5R_TTS_HF_REPO_ID", "").strip()
    if not repo_id:
        return

    if not os.path.exists(local_path):
        return

    token = (
        os.environ.get("F5R_TTS_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    repo_type = os.environ.get("F5R_TTS_HF_REPO_TYPE", "model")
    revision = os.environ.get("F5R_TTS_HF_REVISION")
    subdir = os.environ.get("F5R_TTS_HF_CHECKPOINT_DIR", "checkpoints").strip()

    filename = os.path.basename(local_path)
    path_in_repo = str(Path(subdir) / filename) if subdir else filename

    default_message = os.environ.get("F5R_TTS_HF_COMMIT_TEMPLATE", "Add checkpoint {filename}")
    last_message = os.environ.get("F5R_TTS_HF_COMMIT_LAST") or default_message
    commit_message = (last_message if last else default_message).format(filename=filename, step=step)

    api = HfApi(token=token or None)

    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
        )
    except Exception as err:  # pragma: no cover - relies on external service
        if _is_enabled(os.environ.get("F5R_TTS_HF_VERBOSE", "1")):
            print(f"[HF Upload] Failed to push {filename}: {err}")
        if _is_enabled(os.environ.get("F5R_TTS_HF_STRICT")):
            raise


__all__ = ["push_checkpoint_to_hub"]
