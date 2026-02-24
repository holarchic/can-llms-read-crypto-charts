"""Remote kill switch via GCP Secret Manager flag."""

import logging
import os

from .secrets import get_secret

log = logging.getLogger(__name__)

_TRUTHY = {"true", "1", "yes"}


def is_killed(config) -> bool:
    """Check if kill switch is active. Fail-open (error = not killed).

    Checks GCP SM secret 'KILL_SWITCH' first, then env var fallback.
    """
    try:
        if config.gcp_project_id:
            value = get_secret(config.gcp_project_id, "KILL_SWITCH")
            if value:
                return value.strip().lower() in _TRUTHY
    except Exception as exc:
        log.warning("Error checking kill switch (SM): %s — fail-open", exc)

    # Env var fallback
    env_val = os.getenv("KILL_SWITCH", "false")
    return env_val.strip().lower() in _TRUTHY


def _get_sm_client():
    """Lazy import + instantiate SM client. Returns None if unavailable."""
    try:
        from google.cloud import secretmanager
        return secretmanager.SecretManagerServiceClient()
    except ImportError:
        log.error("google-cloud-secret-manager not installed — cannot set kill switch")
        return None


def set_kill_switch(active: bool, config) -> None:
    """Update kill switch value in GCP Secret Manager.

    Used by CLI kill/unkill commands.
    """
    if not config.gcp_project_id:
        log.error("No GCP project ID configured — cannot set kill switch")
        return

    client = _get_sm_client()
    if client is None:
        return

    try:
        parent = f"projects/{config.gcp_project_id}/secrets/KILL_SWITCH"
        payload = b"true" if active else b"false"
        client.add_secret_version(
            request={"parent": parent, "payload": {"data": payload}}
        )
        log.info("Kill switch %s", "ACTIVATED" if active else "DEACTIVATED")
    except Exception as exc:
        log.error("Failed to set kill switch: %s", exc)
        raise
