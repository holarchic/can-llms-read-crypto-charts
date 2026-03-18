"""GCP Secret Manager integration with env var fallback."""

import logging
import os

log = logging.getLogger(__name__)


def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Fetch a secret from GCP Secret Manager.

    Returns empty string on any failure (missing secret, permission error, etc).
    """
    try:
        from google.cloud import secretmanager
    except ImportError:
        log.debug("google-cloud-secret-manager not installed, skipping SM lookup")
        return ""

    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as exc:
        log.warning("Failed to fetch secret %s: %s", secret_id, exc)
        return ""


def get_secret_or_env(
    secret_id: str, env_key: str, project_id: str = "", *, require: bool = False,
) -> str:
    """Try GCP Secret Manager first, fall back to env var.

    If project_id is empty, skips SM lookup entirely.
    If require=True and SM lookup fails/returns empty, raises RuntimeError
    instead of falling back to env vars (use for live mode credentials).
    """
    if project_id:
        value = get_secret(project_id, secret_id)
        if value:
            return value
        if require:
            raise RuntimeError(
                f"Secret '{secret_id}' unavailable from Secret Manager "
                f"(project={project_id}) and require=True"
            )

    return os.getenv(env_key, "")
