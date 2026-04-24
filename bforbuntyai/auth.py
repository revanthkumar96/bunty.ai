"""HuggingFace Hub authentication helpers.

Usage:
    from bforbuntyai import auth
    auth.login()                    # interactive prompt
    auth.login(token="hf_...")      # explicit token
    auth.whoami()                   # show current user
    auth.logout()

Token resolution order (used by all HF-backed models):
    1. token= argument passed directly to the model
    2. HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable
    3. Token cached by auth.login()
    4. Token saved by `huggingface-cli login`
"""

import os
from typing import Optional

_cached_token: Optional[str] = None


def login(token: Optional[str] = None) -> None:
    global _cached_token
    _require_hub()
    import huggingface_hub

    if token:
        huggingface_hub.login(token=token, add_to_git_credential=False)
        _cached_token = token
    else:
        huggingface_hub.login(add_to_git_credential=False)
        _cached_token = huggingface_hub.get_token()

    try:
        user = huggingface_hub.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("Logged in to HuggingFace Hub.")


def logout() -> None:
    global _cached_token
    _require_hub()
    import huggingface_hub

    huggingface_hub.logout()
    _cached_token = None
    print("Logged out from HuggingFace Hub.")


def whoami() -> dict:
    _require_hub()
    import huggingface_hub

    return huggingface_hub.whoami()


def get_token(token: Optional[str] = None) -> Optional[str]:
    if token:
        return token
    if _cached_token:
        return _cached_token
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env:
        return env
    try:
        import huggingface_hub

        return huggingface_hub.get_token()
    except Exception:
        return None


def require_token(token: Optional[str] = None) -> str:
    t = get_token(token)
    if not t:
        raise ValueError(
            "This model requires a HuggingFace token.\n"
            "  Option 1: from bforbuntyai import auth; auth.login()\n"
            "  Option 2: set the HF_TOKEN environment variable\n"
            "  Option 3: pass token= directly to the model constructor"
        )
    return t


def _require_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auth features.\n"
            "Install with: pip install huggingface_hub"
        )
