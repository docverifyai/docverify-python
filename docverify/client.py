"""
DocVerify Python SDK client.

Synchronous client using httpx. Handles authentication, error mapping,
and response parsing.

Usage:
    client = DocVerifyClient(api_key="dv_...")
    result = client.verify(source="...", ai_output="...")
"""

from pathlib import Path
from typing import Literal

import httpx

from docverify.errors import AuthenticationError, DocVerifyError, RateLimitError, ValidationError
from docverify.models import (
    HealthResponse,
    VerificationResponse,
    _parse_health_response,
    _parse_verification_response,
)

DEFAULT_BASE_URL = "https://api.docverify.dev"
DEFAULT_TIMEOUT = 120.0  # Model inference can be slow


class DocVerifyClient:
    """
    DocVerify API client.

    Args:
        api_key: Your API key (starts with "dv_")
        base_url: API base URL (default: https://api.docverify.dev)
        timeout: Request timeout in seconds (default: 120)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key.startswith("dv_"):
            raise ValueError("API key must start with 'dv_'")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "docverify-python/1.0.0",
            },
            timeout=timeout,
        )

    def verify(
        self,
        source: str,
        ai_output: str,
        source_type: Literal["text", "pdf", "docx"] = "text",
        task_type: Literal["extraction", "summary", "analysis"] = "extraction",
        domain: Literal["legal", "financial", "medical", "general"] = "general",
        return_evidence: bool = True,
        severity_threshold: Literal["low", "medium", "high"] = "low",
    ) -> VerificationResponse:
        """
        Verify AI-generated output against a source document.

        Args:
            source: Source document text (or base64 for pdf/docx)
            ai_output: The AI-generated text to verify
            source_type: Format of the source document
            task_type: What the AI was asked to do
            domain: Domain for specialized NLI model
            return_evidence: Include evidence citations
            severity_threshold: Minimum severity to report

        Returns:
            VerificationResponse with claim-by-claim results
        """
        payload = {
            "source": source,
            "source_type": source_type,
            "ai_output": ai_output,
            "task_type": task_type,
            "domain": domain,
            "options": {
                "return_evidence": return_evidence,
                "severity_threshold": severity_threshold,
            },
        }
        response = self._request("POST", "/v1/verify", json=payload)
        return _parse_verification_response(response)

    def verify_file(
        self,
        file_path: str | Path,
        ai_output: str,
        task_type: Literal["extraction", "summary", "analysis"] = "extraction",
        domain: Literal["legal", "financial", "medical", "general"] = "general",
    ) -> VerificationResponse:
        """
        Verify AI output against a file (PDF, DOCX, or TXT).

        Uploads the file as multipart form data to /v1/verify/file.

        Args:
            file_path: Path to the source document
            ai_output: The AI-generated text to verify
            task_type: What the AI was asked to do
            domain: Domain for specialized NLI model

        Returns:
            VerificationResponse with claim-by-claim results
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in (".pdf", ".docx", ".doc", ".txt"):
            raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .docx, or .txt")

        content = path.read_bytes()

        # Determine content type
        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
        }
        content_type = content_types.get(suffix, "application/octet-stream")

        # Send as multipart form data
        files = {"file": (path.name, content, content_type)}
        data = {
            "ai_output": ai_output,
            "task_type": task_type,
            "domain": domain,
        }

        try:
            response = self._client.post(
                "/v1/verify/file",
                files=files,
                data=data,
                headers={},  # Let httpx set Content-Type for multipart
            )
        except httpx.ConnectError as e:
            raise DocVerifyError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise DocVerifyError(f"Request timed out: {e}") from e

        if response.status_code in (200, 201):
            return _parse_verification_response(response.json())

        detail = ""
        try:
            error_data = response.json()
            detail = error_data.get("detail", str(error_data))
        except Exception:
            detail = response.text

        if response.status_code == 401:
            raise AuthenticationError(detail)
        if response.status_code == 413:
            raise ValidationError(f"File too large: {detail}")
        if response.status_code == 422:
            raise ValidationError(detail)
        if response.status_code == 429:
            raise RateLimitError(detail)

        raise DocVerifyError(detail, status_code=response.status_code)

    def verify_stream(
        self,
        source: str,
        ai_output: str,
        source_type: Literal["text", "pdf", "docx"] = "text",
        task_type: Literal["extraction", "summary", "analysis"] = "extraction",
        domain: Literal["legal", "financial", "medical", "general"] = "general",
    ):
        """
        Stream verification results via Server-Sent Events.

        Returns a generator that yields SSE event dicts as they arrive.
        Each event has an 'event_type' field: 'status', 'claim', 'complete', or 'error'.

        Args:
            source: Source document text (or base64 for pdf/docx)
            ai_output: The AI-generated text to verify
            source_type: Format of the source document
            task_type: What the AI was asked to do
            domain: Domain for specialized NLI model

        Yields:
            dict with event_type and event-specific data
        """
        import json

        payload = {
            "source": source,
            "source_type": source_type,
            "ai_output": ai_output,
            "task_type": task_type,
            "domain": domain,
        }

        try:
            with self._client.stream("POST", "/v1/verify/stream", json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    detail = response.text
                    try:
                        detail = response.json().get("detail", detail)
                    except Exception:
                        pass
                    if response.status_code == 401:
                        raise AuthenticationError(detail)
                    if response.status_code == 429:
                        raise RateLimitError(detail)
                    raise DocVerifyError(detail, status_code=response.status_code)

                event_type = "status"
                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event_type = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        try:
                            data = json.loads(data_str)
                            data["event_type"] = event_type
                            yield data
                        except json.JSONDecodeError:
                            yield {"event_type": event_type, "raw": data_str}
        except httpx.ConnectError as e:
            raise DocVerifyError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise DocVerifyError(f"Request timed out: {e}") from e

    def submit_feedback(
        self,
        verification_id: str,
        claim_index: int,
        corrected_verdict: Literal[
            "SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "PARTIAL", "AMBIGUOUS"
        ],
        notes: str | None = None,
    ) -> dict:
        """
        Submit verdict correction feedback.

        Args:
            verification_id: The verification ID from a previous verify() call
            claim_index: Index of the claim to correct (0-based)
            corrected_verdict: What the verdict should be
            notes: Optional explanation

        Returns:
            Dict with feedback_id and status
        """
        payload: dict = {
            "verification_id": verification_id,
            "claim_index": claim_index,
            "corrected_verdict": corrected_verdict,
        }
        if notes is not None:
            payload["notes"] = notes
        return self._request("POST", "/v1/feedback", json=payload)

    def get_feedback(
        self,
        verification_id: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """
        Get feedback items for the authenticated API key.

        Args:
            verification_id: Filter by verification ID
            status: Filter by status (pending, accepted, rejected)

        Returns:
            List of feedback items
        """
        params: dict = {}
        if verification_id is not None:
            params["verification_id"] = verification_id
        if status is not None:
            params["status"] = status
        response = self._request("GET", "/v1/feedback", params=params)
        return response.get("feedback", [])

    def batch_verify(
        self,
        source: str,
        ai_outputs: list[dict],
        source_type: Literal["text", "pdf", "docx"] = "text",
        domain: Literal["legal", "financial", "medical", "general"] = "general",
    ) -> list[dict]:
        """
        Verify multiple AI outputs against a single source document.

        Args:
            source: Source document text (or base64 for pdf/docx)
            ai_outputs: List of dicts with keys: id, text, and optional task_type
            source_type: Format of the source document
            domain: Domain for specialized NLI model

        Returns:
            List of verification result dicts
        """
        payload = {
            "source": source,
            "source_type": source_type,
            "ai_outputs": ai_outputs,
            "domain": domain,
        }
        response = self._request("POST", "/v1/verify/batch", json=payload)
        return response.get("results", [])

    def compare(
        self,
        source: str,
        ai_outputs: list[dict],
        source_type: Literal["text", "pdf", "docx"] = "text",
        domain: Literal["legal", "financial", "medical", "general"] = "general",
    ) -> dict:
        """
        Compare multiple AI outputs against a single source document.

        Runs batch verification then cross-compares claims across outputs,
        identifying agreements and disagreements.

        Args:
            source: Source document text (or base64 for pdf/docx)
            ai_outputs: List of dicts with keys: id, text, and optional task_type
            source_type: Format of the source document
            domain: Domain for specialized NLI model

        Returns:
            Dict with results (list), comparison (list), and summary (dict)
        """
        payload = {
            "source": source,
            "source_type": source_type,
            "ai_outputs": ai_outputs,
            "domain": domain,
        }
        return self._request("POST", "/v1/verify/compare", json=payload)

    def create_adapter(
        self,
        name: str,
        training_data: list[dict],
        description: str | None = None,
    ) -> dict:
        """
        Create a custom LoRA adapter from labeled NLI data.

        Args:
            name: Adapter name
            training_data: List of {premise, hypothesis, label} dicts
            description: Optional description

        Returns:
            Dict with adapter_id, name, status, training_samples, message
        """
        payload: dict = {"name": name, "training_data": training_data}
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/v1/adapters", json=payload)

    def list_adapters(self) -> list[dict]:
        """
        List custom adapters for the authenticated API key.

        Returns:
            List of adapter dicts
        """
        response = self._request("GET", "/v1/adapters")
        return response.get("adapters", [])

    def delete_adapter(self, adapter_id: str) -> dict:
        """
        Delete a custom adapter.

        Args:
            adapter_id: The adapter ID (e.g. "adp_abc123def456")

        Returns:
            Dict with status and adapter_id
        """
        return self._request("DELETE", f"/v1/adapters/{adapter_id}")

    def dashboard_stats(self) -> dict:
        """
        Get dashboard statistics for the authenticated API key.

        Returns:
            Dict with total_verifications, verdict_counts, average_score,
            average_processing_time_ms, recent_verifications
        """
        return self._request("GET", "/v1/dashboard/stats")

    def diagnostics(self) -> dict:
        """
        Get comprehensive system diagnostics.

        Returns:
            Dict with status, checks (database, redis, pipeline, disk, memory, gpu),
            and versions (api, python, torch, transformers)
        """
        return self._request("GET", "/v1/diagnostics")

    def audit_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        action: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """
        Get audit log entries for the authenticated API key.

        Args:
            limit: Max results per page (1-200, default 50)
            offset: Pagination offset (default 0)
            action: Filter by action type (e.g. "verify", "feedback_submit")
            start_date: Filter events after this ISO date
            end_date: Filter events before this ISO date

        Returns:
            Dict with events (list), total (int), limit (int), offset (int)
        """
        params: dict = {"limit": limit, "offset": offset}
        if action is not None:
            params["action"] = action
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        return self._request("GET", "/v1/audit/logs", params=params)

    def get_report(self, verification_id: str) -> bytes:
        """
        Download a PDF verification report.

        Args:
            verification_id: The verification ID to generate a report for

        Returns:
            PDF file contents as bytes
        """
        try:
            response = self._client.request("GET", f"/v1/reports/{verification_id}")
        except httpx.ConnectError as e:
            raise DocVerifyError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise DocVerifyError(f"Request timed out: {e}") from e

        if response.status_code == 200:
            return response.content

        detail = ""
        try:
            error_data = response.json()
            detail = error_data.get("detail", str(error_data))
        except Exception:
            detail = response.text

        if response.status_code == 401:
            raise AuthenticationError(detail)
        if response.status_code == 404:
            raise DocVerifyError(f"Verification not found: {detail}", status_code=404)
        raise DocVerifyError(detail, status_code=response.status_code)

    def subscribe_webhook(self, url: str, events: list[str], secret: str) -> dict:
        """
        Subscribe a webhook URL for event notifications.

        Args:
            url: HTTPS URL to receive webhook POST requests
            events: List of event types to subscribe to
            secret: Shared secret for HMAC-SHA256 payload signing (min 16 chars)

        Returns:
            Dict with webhook_id, url, events, status
        """
        return self._request(
            "POST",
            "/v1/webhooks/subscribe",
            json={
                "url": url,
                "events": events,
                "secret": secret,
            },
        )

    def list_webhooks(self) -> dict:
        """
        List active webhook subscriptions for the authenticated API key.

        Returns:
            Dict with webhooks (list of subscription dicts)
        """
        return self._request("GET", "/v1/webhooks")

    def delete_webhook(self, webhook_id: str) -> dict:
        """
        Delete a webhook subscription.

        Args:
            webhook_id: The webhook ID to delete (e.g. "wh_abc123...")

        Returns:
            Dict with webhook_id and status
        """
        return self._request("DELETE", f"/v1/webhooks/{webhook_id}")

    def rotate_key(self) -> dict:
        """
        Rotate the current API key.

        Returns:
            Dict with new_api_key, new_key_prefix, old_key_status, grace_period_hours
        """
        return self._request("POST", "/v1/keys/rotate")

    def get_scopes(self) -> dict:
        """
        Get the current scopes for the authenticated API key.

        Returns:
            Dict with api_key_id and scopes (list)
        """
        return self._request("GET", "/v1/keys/scopes")

    def update_scopes(self, scopes: list[str]) -> dict:
        """
        Update scopes for the authenticated API key.

        Args:
            scopes: List of scope strings (e.g. ["verify", "feedback", "admin"])

        Returns:
            Dict with api_key_id and updated scopes
        """
        return self._request("PUT", "/v1/keys/scopes", json={"scopes": scopes})

    def create_team(self, name: str, description: str | None = None) -> dict:
        """
        Create a new team.

        Args:
            name: Team name
            description: Optional description

        Returns:
            Dict with team_id, name, description, created_at
        """
        payload: dict = {"name": name}
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/v1/teams", json=payload)

    def list_teams(self) -> dict:
        """
        List teams the current API key belongs to.

        Returns:
            Dict with teams (list)
        """
        return self._request("GET", "/v1/teams")

    def get_team(self, team_id: str) -> dict:
        """
        Get team details.

        Args:
            team_id: The team ID

        Returns:
            Dict with team_id, name, description, member_count, created_at
        """
        return self._request("GET", f"/v1/teams/{team_id}")

    def add_team_member(self, team_id: str, email: str, role: str = "member") -> dict:
        """
        Add a member to a team.

        Args:
            team_id: The team ID
            email: Member's email address
            role: Role (admin, member, or viewer)

        Returns:
            Dict with member_id, team_id, email, role
        """
        return self._request(
            "POST", f"/v1/teams/{team_id}/members", json={"email": email, "role": role}
        )

    def list_team_members(self, team_id: str) -> dict:
        """
        List members of a team.

        Args:
            team_id: The team ID

        Returns:
            Dict with members (list)
        """
        return self._request("GET", f"/v1/teams/{team_id}/members")

    def remove_team_member(self, team_id: str, member_id: str) -> dict:
        """
        Remove a member from a team.

        Args:
            team_id: The team ID
            member_id: The member ID to remove

        Returns:
            Dict with status
        """
        return self._request("DELETE", f"/v1/teams/{team_id}/members/{member_id}")

    def get_ip_allowlist(self) -> dict:
        """
        Get the current IP allowlist for the authenticated API key.

        Returns:
            Dict with api_key_id, allowed_ips, enabled
        """
        return self._request("GET", "/v1/keys/ip-allowlist")

    def set_ip_allowlist(self, allowed_ips: list[str]) -> dict:
        """
        Set the IP allowlist for the authenticated API key.

        Args:
            allowed_ips: List of IPs or CIDRs (e.g. ["1.2.3.4", "10.0.0.0/8"])

        Returns:
            Dict with api_key_id, allowed_ips, enabled
        """
        return self._request("PUT", "/v1/keys/ip-allowlist", json={"allowed_ips": allowed_ips})

    def clear_ip_allowlist(self) -> dict:
        """
        Clear the IP allowlist (disable enforcement).

        Returns:
            Dict with api_key_id and status
        """
        return self._request("DELETE", "/v1/keys/ip-allowlist")

    def export_audit(
        self,
        format: str = "json",
        start_date: str | None = None,
        end_date: str | None = None,
        action: str | None = None,
    ) -> bytes | dict:
        """
        Export audit logs.

        Args:
            format: "json" or "csv"
            start_date: Filter events after this ISO date
            end_date: Filter events before this ISO date
            action: Filter by action type

        Returns:
            Dict (JSON) or bytes (CSV)
        """
        params: dict = {"format": format}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if action is not None:
            params["action"] = action

        if format == "csv":
            try:
                response = self._client.request("GET", "/v1/export/audit", params=params)
            except httpx.ConnectError as e:
                raise DocVerifyError(f"Connection failed: {e}") from e
            except httpx.TimeoutException as e:
                raise DocVerifyError(f"Request timed out: {e}") from e
            if response.status_code == 200:
                return response.content
            raise DocVerifyError(response.text, status_code=response.status_code)

        return self._request("GET", "/v1/export/audit", params=params)

    def export_verifications(
        self,
        format: str = "json",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> bytes | dict:
        """
        Export verification history.

        Args:
            format: "json" or "csv"
            start_date: Filter after this ISO date
            end_date: Filter before this ISO date

        Returns:
            Dict (JSON) or bytes (CSV)
        """
        params: dict = {"format": format}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date

        if format == "csv":
            try:
                response = self._client.request("GET", "/v1/export/verifications", params=params)
            except httpx.ConnectError as e:
                raise DocVerifyError(f"Connection failed: {e}") from e
            except httpx.TimeoutException as e:
                raise DocVerifyError(f"Request timed out: {e}") from e
            if response.status_code == 200:
                return response.content
            raise DocVerifyError(response.text, status_code=response.status_code)

        return self._request("GET", "/v1/export/verifications", params=params)

    def get_analytics(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        group_by: str = "day",
    ) -> dict:
        """
        Get usage analytics.

        Args:
            start_date: Filter after this ISO date
            end_date: Filter before this ISO date
            group_by: Breakdown key ("day", "domain", or "verdict")

        Returns:
            Dict with period, total_verifications, breakdown, top_verdicts, avg_processing_time_ms
        """
        params: dict = {"group_by": group_by}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        return self._request("GET", "/v1/analytics", params=params)

    def list_templates(self) -> dict:
        """List all verification templates (built-in + custom)."""
        return self._request("GET", "/v1/templates")

    def create_template(
        self,
        name: str,
        domain: str = "general",
        task_type: str = "extraction",
        options: dict | None = None,
        description: str | None = None,
    ) -> dict:
        """Create a custom verification template."""
        payload: dict = {"name": name, "domain": domain, "task_type": task_type}
        if options is not None:
            payload["options"] = options
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/v1/templates", json=payload)

    def get_template(self, template_id: str) -> dict:
        """Get a template by ID."""
        return self._request("GET", f"/v1/templates/{template_id}")

    def delete_template(self, template_id: str) -> dict:
        """Delete a custom template."""
        return self._request("DELETE", f"/v1/templates/{template_id}")

    def verify_with_template(
        self,
        template_id: str,
        source: str,
        ai_output: str,
        source_type: str = "text",
    ) -> dict:
        """Verify AI output using a template's preset configuration."""
        payload = {"source": source, "ai_output": ai_output, "source_type": source_type}
        return self._request("POST", f"/v1/verify/template/{template_id}", json=payload)

    def get_retention(self) -> dict:
        """Get the current data retention policy."""
        return self._request("GET", "/v1/retention")

    def set_retention(
        self,
        audit_days: int | None = None,
        verification_days: int | None = None,
        feedback_days: int | None = None,
    ) -> dict:
        """Set data retention policy."""
        payload: dict = {}
        if audit_days is not None:
            payload["audit_retention_days"] = audit_days
        if verification_days is not None:
            payload["verification_retention_days"] = verification_days
        if feedback_days is not None:
            payload["feedback_retention_days"] = feedback_days
        return self._request("PUT", "/v1/retention", json=payload)

    def apply_retention(self) -> dict:
        """Manually trigger retention cleanup."""
        return self._request("POST", "/v1/retention/apply")

    def clear_retention(self) -> dict:
        """Remove retention policy (disable auto-cleanup)."""
        return self._request("DELETE", "/v1/retention")

    def list_namespaces(self) -> dict:
        """List all namespaces for the authenticated API key."""
        return self._request("GET", "/v1/namespaces")

    def create_namespace(self, name: str, description: str | None = None) -> dict:
        """Create a new namespace."""
        payload: dict = {"name": name}
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/v1/namespaces", json=payload)

    def delete_namespace(self, namespace_id: str) -> dict:
        """Delete a namespace."""
        return self._request("DELETE", f"/v1/namespaces/{namespace_id}")

    def get_explanation(self, verification_id: str) -> dict:
        """Get detailed explanations for a verification result."""
        return self._request("GET", f"/v1/verify/{verification_id}/explain")

    def submit_job(
        self,
        job_type: str,
        source: str,
        ai_outputs: list[dict],
        source_type: str = "text",
        domain: str = "general",
    ) -> dict:
        """Submit an async verification job."""
        payload: dict = {
            "job_type": job_type,
            "source": source,
            "ai_outputs": ai_outputs,
            "source_type": source_type,
            "domain": domain,
        }
        return self._request("POST", "/v1/jobs", json=payload)

    def get_job(self, job_id: str) -> dict:
        """Get job detail and results when completed."""
        return self._request("GET", f"/v1/jobs/{job_id}")

    def list_jobs(self) -> dict:
        """List jobs for the authenticated API key."""
        return self._request("GET", "/v1/jobs")

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a pending job."""
        return self._request("DELETE", f"/v1/jobs/{job_id}")

    def get_webhook_deliveries(self, webhook_id: str) -> dict:
        """Get delivery attempts for a webhook subscription."""
        return self._request("GET", f"/v1/webhooks/{webhook_id}/deliveries")

    def retry_webhook_delivery(self, webhook_id: str, delivery_id: str) -> dict:
        """Retry a failed webhook delivery."""
        return self._request("POST", f"/v1/webhooks/{webhook_id}/deliveries/{delivery_id}/retry")

    def create_alert(
        self,
        name: str,
        metric: str,
        threshold: float,
        comparison: str = "gte",
    ) -> dict:
        """Create a usage alert rule."""
        payload: dict = {
            "name": name,
            "metric": metric,
            "threshold": threshold,
            "comparison": comparison,
        }
        return self._request("POST", "/v1/alerts", json=payload)

    def list_alerts(self) -> dict:
        """List active alert rules."""
        return self._request("GET", "/v1/alerts")

    def delete_alert(self, alert_id: str) -> dict:
        """Delete an alert rule."""
        return self._request("DELETE", f"/v1/alerts/{alert_id}")

    def get_alert_history(self) -> dict:
        """List triggered alert events."""
        return self._request("GET", "/v1/alerts/history")

    def create_policy(
        self,
        name: str,
        action: str,
        conditions: list[dict],
        description: str | None = None,
        enabled: bool = True,
    ) -> dict:
        """Create a verification policy."""
        payload: dict = {
            "name": name,
            "action": action,
            "conditions": conditions,
            "enabled": enabled,
        }
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/v1/policies", json=payload)

    def list_policies(self) -> dict:
        """List policies for the authenticated API key."""
        return self._request("GET", "/v1/policies")

    def get_policy(self, policy_id: str) -> dict:
        """Get policy detail."""
        return self._request("GET", f"/v1/policies/{policy_id}")

    def update_policy(self, policy_id: str, **kwargs) -> dict:
        """Update a policy."""
        return self._request("PUT", f"/v1/policies/{policy_id}", json=kwargs)

    def delete_policy(self, policy_id: str) -> dict:
        """Delete a policy."""
        return self._request("DELETE", f"/v1/policies/{policy_id}")

    def list_languages(self) -> dict:
        """List supported languages with status."""
        return self._request("GET", "/v1/languages")

    def detect_language(self, text: str) -> dict:
        """Detect language from text."""
        return self._request("GET", "/v1/languages/detect", params={"text": text})

    def get_language_config(self) -> dict:
        """Get language preferences for the authenticated API key."""
        return self._request("GET", "/v1/languages/config")

    def set_language_config(self, **kwargs) -> dict:
        """Set language preferences."""
        return self._request("PUT", "/v1/languages/config", json=kwargs)

    def list_model_versions(self) -> dict:
        """List registered model versions."""
        return self._request("GET", "/v1/models/versions")

    def create_model_version(
        self, name: str, model_path: str | None = None, metrics: dict | None = None
    ) -> dict:
        """Register a new model version."""
        payload: dict = {"name": name}
        if model_path is not None:
            payload["model_path"] = model_path
        if metrics is not None:
            payload["metrics"] = metrics
        return self._request("POST", "/v1/models/versions", json=payload)

    def get_model_version(self, version_id: str) -> dict:
        """Get model version detail."""
        return self._request("GET", f"/v1/models/versions/{version_id}")

    def retire_model_version(self, version_id: str) -> dict:
        """Retire a model version."""
        return self._request("DELETE", f"/v1/models/versions/{version_id}")

    def create_ab_test(
        self,
        name: str,
        control_version: str,
        treatment_version: str,
        traffic_percent: int,
    ) -> dict:
        """Create an A/B test between two model versions."""
        return self._request(
            "POST",
            "/v1/models/ab-tests",
            json={
                "name": name,
                "control_version": control_version,
                "treatment_version": treatment_version,
                "traffic_percent": traffic_percent,
            },
        )

    def list_ab_tests(self) -> dict:
        """List A/B tests."""
        return self._request("GET", "/v1/models/ab-tests")

    def get_ab_test(self, test_id: str) -> dict:
        """Get A/B test detail with results."""
        return self._request("GET", f"/v1/models/ab-tests/{test_id}")

    def cancel_ab_test(self, test_id: str) -> dict:
        """Cancel an A/B test."""
        return self._request("DELETE", f"/v1/models/ab-tests/{test_id}")

    def get_cache_stats(self) -> dict:
        """Get semantic cache statistics."""
        return self._request("GET", "/v1/cache/stats")

    def clear_cache(self) -> dict:
        """Clear semantic cache for the authenticated API key."""
        return self._request("DELETE", "/v1/cache")

    def get_cache_config(self) -> dict:
        """Get cache settings."""
        return self._request("GET", "/v1/cache/config")

    def set_cache_config(self, **kwargs) -> dict:
        """Update cache settings."""
        return self._request("PUT", "/v1/cache/config", json=kwargs)

    def get_webhook_stats(self, webhook_id: str) -> dict:
        """Get delivery statistics for a webhook."""
        return self._request("GET", f"/v1/webhooks/{webhook_id}/stats")

    def list_dead_letter(self, webhook_id: str) -> dict:
        """List dead letter queue entries for a webhook."""
        return self._request("GET", f"/v1/webhooks/{webhook_id}/dead-letter")

    def replay_dead_letter(self, webhook_id: str, entry_id: str) -> dict:
        """Re-enqueue a dead letter entry for retry."""
        return self._request("POST", f"/v1/webhooks/{webhook_id}/dead-letter/{entry_id}/replay")

    def dismiss_dead_letter(self, webhook_id: str, entry_id: str) -> dict:
        """Dismiss a dead letter entry."""
        return self._request("DELETE", f"/v1/webhooks/{webhook_id}/dead-letter/{entry_id}")

    def get_compliance_status(self) -> dict:
        """Get comprehensive compliance posture."""
        return self._request("GET", "/v1/compliance/status")

    def get_compliance_report(self) -> dict:
        """Get full compliance report."""
        return self._request("GET", "/v1/compliance/report")

    def get_security_headers(self) -> dict:
        """List security headers applied to all API responses."""
        return self._request("GET", "/v1/security/headers")

    def get_security_config(self) -> dict:
        """Get security settings."""
        return self._request("GET", "/v1/security/config")

    def set_security_config(self, **kwargs) -> dict:
        """Update security settings."""
        return self._request("PUT", "/v1/security/config", json=kwargs)

    def search_verifications(self, **kwargs) -> dict:
        """Search verification history with filters."""
        return self._request("GET", "/v1/search/verifications", params=kwargs)

    def search_claims(self, **kwargs) -> dict:
        """Search individual claims across verifications."""
        return self._request("GET", "/v1/search/claims", params=kwargs)

    def get_verification_profile(self, verification_id: str) -> dict:
        """Get timing breakdown for a verification."""
        return self._request("GET", f"/v1/verify/{verification_id}/profile")

    def get_performance_analytics(self, **kwargs) -> dict:
        """Get aggregate performance statistics."""
        return self._request("GET", "/v1/analytics/performance", params=kwargs)

    def add_claim_tag(self, verification_id: str, tag: str, claim_index: int = 0) -> dict:
        """Add a tag to a claim."""
        return self._request(
            "POST",
            f"/v1/claims/{verification_id}/tags",
            json={"tag": tag, "claim_index": claim_index},
        )

    def list_claim_tags(self, verification_id: str) -> dict:
        """List tags for a verification."""
        return self._request("GET", f"/v1/claims/{verification_id}/tags")

    def delete_claim_tag(self, verification_id: str, tag_id: str) -> dict:
        """Delete a claim tag."""
        return self._request("DELETE", f"/v1/claims/{verification_id}/tags/{tag_id}")

    def add_claim_annotation(
        self, verification_id: str, text: str, claim_index: int = 0, type: str = "note"
    ) -> dict:
        """Add an annotation to a claim."""
        return self._request(
            "POST",
            f"/v1/claims/{verification_id}/annotations",
            json={"claim_index": claim_index, "type": type, "text": text},
        )

    def list_claim_annotations(self, verification_id: str) -> dict:
        """List annotations for a verification."""
        return self._request("GET", f"/v1/claims/{verification_id}/annotations")

    def delete_claim_annotation(self, verification_id: str, annotation_id: str) -> dict:
        """Delete a claim annotation."""
        return self._request("DELETE", f"/v1/claims/{verification_id}/annotations/{annotation_id}")

    def create_snapshot(self, name: str, verification_id: str, description: str = "") -> dict:
        """Create a snapshot from a verification result."""
        return self._request(
            "POST",
            "/v1/snapshots",
            json={"name": name, "verification_id": verification_id, "description": description},
        )

    def list_snapshots(self, **kwargs) -> dict:
        """List snapshots."""
        return self._request("GET", "/v1/snapshots", params=kwargs)

    def get_snapshot(self, snapshot_id: str) -> dict:
        """Get snapshot detail."""
        return self._request("GET", f"/v1/snapshots/{snapshot_id}")

    def delete_snapshot(self, snapshot_id: str) -> dict:
        """Delete a snapshot."""
        return self._request("DELETE", f"/v1/snapshots/{snapshot_id}")

    def diff_snapshots(self, snapshot_a: str, snapshot_b: str) -> dict:
        """Compare two snapshots."""
        return self._request("GET", "/v1/snapshots/diff", params={"a": snapshot_a, "b": snapshot_b})

    def get_usage_forecast(self, days: int = 7) -> dict:
        """Predict future usage."""
        return self._request("GET", "/v1/analytics/forecast", params={"days": days})

    def get_usage_trends(self, period_days: int = 30) -> dict:
        """Analyze usage patterns."""
        return self._request("GET", "/v1/analytics/trends", params={"period_days": period_days})

    def set_verification_status(
        self, verification_id: str, status: str, comment: str | None = None
    ) -> dict:
        """Transition verification workflow status."""
        payload: dict = {"status": status}
        if comment is not None:
            payload["comment"] = comment
        return self._request("POST", f"/v1/verifications/{verification_id}/status", json=payload)

    def get_verification_status(self, verification_id: str) -> dict:
        """Get current workflow status and history."""
        return self._request("GET", f"/v1/verifications/{verification_id}/status")

    def add_sign_off(
        self, verification_id: str, signer_name: str, comment: str | None = None
    ) -> dict:
        """Add a sign-off to a verification."""
        payload: dict = {"signer_name": signer_name}
        if comment is not None:
            payload["comment"] = comment
        return self._request("POST", f"/v1/verifications/{verification_id}/sign-off", json=payload)

    def list_sign_offs(self, verification_id: str) -> dict:
        """List sign-offs for a verification."""
        return self._request("GET", f"/v1/verifications/{verification_id}/sign-offs")

    def estimate_cost(
        self, source_length: int, claim_count: int = 0, domain: str = "general"
    ) -> dict:
        """Estimate cost for a verification."""
        return self._request(
            "POST",
            "/v1/costs/estimate",
            json={"source_length": source_length, "claim_count": claim_count, "domain": domain},
        )

    def get_cost_summary(self) -> dict:
        """Get spending summary from verification history."""
        return self._request("GET", "/v1/costs/summary")

    def create_budget(
        self, name: str, limit_usd: float, period: str, domain: str | None = None
    ) -> dict:
        """Create a budget."""
        payload: dict = {"name": name, "limit_usd": limit_usd, "period": period}
        if domain is not None:
            payload["domain"] = domain
        return self._request("POST", "/v1/budgets", json=payload)

    def list_budgets(self) -> dict:
        """List budgets."""
        return self._request("GET", "/v1/budgets")

    def delete_budget(self, budget_id: str) -> dict:
        """Delete a budget."""
        return self._request("DELETE", f"/v1/budgets/{budget_id}")

    def get_quality_analytics(self) -> dict:
        """Get overall quality statistics."""
        return self._request("GET", "/v1/analytics/quality")

    def get_quality_by_domain(self) -> dict:
        """Get per-domain quality breakdown."""
        return self._request("GET", "/v1/analytics/quality/by-domain")

    def get_quality_claims(self) -> dict:
        """Get claim-level quality analytics."""
        return self._request("GET", "/v1/analytics/quality/claims")

    def get_recommendations(self) -> dict:
        """Get heuristic improvement recommendations."""
        return self._request("GET", "/v1/analytics/recommendations")

    def set_drift_baseline(self, **kwargs) -> dict:
        """Set drift detection baseline."""
        return self._request("POST", "/v1/monitoring/baseline", json=kwargs)

    def get_drift_analysis(self, **kwargs) -> dict:
        """Get drift analysis against stored baseline."""
        return self._request("GET", "/v1/monitoring/drift", params=kwargs)

    def get_anomalies(self) -> dict:
        """Get anomaly detection results."""
        return self._request("GET", "/v1/monitoring/anomalies")

    def get_health_score(self) -> dict:
        """Get composite model health score (0-100)."""
        return self._request("GET", "/v1/monitoring/health-score")

    def get_provenance(self, verification_id: str) -> dict:
        """Get data lineage for a verification."""
        return self._request("GET", f"/v1/provenance/{verification_id}")

    def gdpr_export(self) -> dict:
        """Export all data for the authenticated API key (GDPR)."""
        return self._request("GET", "/v1/compliance/gdpr/export")

    def gdpr_forget(self) -> dict:
        """Right to be forgotten — clear all data."""
        return self._request("DELETE", "/v1/compliance/gdpr/forget")

    def get_certifications(self) -> dict:
        """Get compliance certification self-assessments."""
        return self._request("GET", "/v1/compliance/certifications")

    def get_rate_limits(self) -> dict:
        """View effective rate limits for caller."""
        return self._request("GET", "/v1/rate-limits")

    def set_rate_limits(self, **kwargs) -> dict:
        """Set custom rate limit overrides."""
        return self._request("PUT", "/v1/rate-limits", json=kwargs)

    def reset_rate_limits(self) -> dict:
        """Reset rate limits to tier defaults."""
        return self._request("DELETE", "/v1/rate-limits")

    def get_rate_limit_status(self) -> dict:
        """Get real-time rate limit usage status."""
        return self._request("GET", "/v1/rate-limits/status")

    def create_sandbox(self, name: str, **kwargs) -> dict:
        """Create a sandbox environment."""
        return self._request("POST", "/v1/sandboxes", json={"name": name, **kwargs})

    def list_sandboxes(self) -> dict:
        """List sandbox environments."""
        return self._request("GET", "/v1/sandboxes")

    def get_sandbox(self, sandbox_id: str) -> dict:
        """Get sandbox detail."""
        return self._request("GET", f"/v1/sandboxes/{sandbox_id}")

    def reset_sandbox(self, sandbox_id: str) -> dict:
        """Reset sandbox data."""
        return self._request("POST", f"/v1/sandboxes/{sandbox_id}/reset")

    def delete_sandbox(self, sandbox_id: str) -> dict:
        """Delete a sandbox."""
        return self._request("DELETE", f"/v1/sandboxes/{sandbox_id}")

    def create_extraction_rule(self, **kwargs) -> dict:
        """Create a custom extraction rule."""
        return self._request("POST", "/v1/extraction-rules", json=kwargs)

    def list_extraction_rules(self, **kwargs) -> dict:
        """List extraction rules."""
        return self._request("GET", "/v1/extraction-rules", params=kwargs)

    def update_extraction_rule(self, rule_id: str, **kwargs) -> dict:
        """Update an extraction rule."""
        return self._request("PUT", f"/v1/extraction-rules/{rule_id}", json=kwargs)

    def delete_extraction_rule(self, rule_id: str) -> dict:
        """Delete an extraction rule."""
        return self._request("DELETE", f"/v1/extraction-rules/{rule_id}")

    def test_extraction_rules(self, text: str, rule_ids: list[str] | None = None) -> dict:
        """Test extraction rules against text."""
        payload: dict = {"text": text}
        if rule_ids is not None:
            payload["rule_ids"] = rule_ids
        return self._request("POST", "/v1/extraction-rules/test", json=payload)

    def get_changelog(self, **kwargs) -> dict:
        """List changelog entries."""
        return self._request("GET", "/v1/changelog", params=kwargs)

    def create_changelog_entry(self, **kwargs) -> dict:
        """Create a changelog entry (admin)."""
        return self._request("POST", "/v1/changelog", json=kwargs)

    def list_deprecations(self, **kwargs) -> dict:
        """List deprecation notices."""
        return self._request("GET", "/v1/deprecations", params=kwargs)

    def create_deprecation(self, **kwargs) -> dict:
        """Create a deprecation notice (admin)."""
        return self._request("POST", "/v1/deprecations", json=kwargs)

    def acknowledge_deprecation(self, notice_id: str) -> dict:
        """Acknowledge a deprecation notice."""
        return self._request("POST", f"/v1/deprecations/{notice_id}/acknowledge")

    def list_classification_labels(self) -> dict:
        """List all classification labels (builtin + custom)."""
        return self._request("GET", "/v1/classifications/labels")

    def create_classification_label(self, **kwargs) -> dict:
        """Create a custom classification label."""
        return self._request("POST", "/v1/classifications/labels", json=kwargs)

    def assign_classification(self, verification_id: str, label: str, **kwargs) -> dict:
        """Assign a classification label to a verification."""
        return self._request(
            "POST",
            "/v1/classifications/assign",
            json={"verification_id": verification_id, "label": label, **kwargs},
        )

    def get_classification(self, verification_id: str) -> dict:
        """Get classification for a verification."""
        return self._request("GET", f"/v1/classifications/{verification_id}")

    def get_classification_summary(self) -> dict:
        """Get classification summary counts."""
        return self._request("GET", "/v1/classifications/summary")

    def get_health(self) -> HealthResponse:
        """Check API health status."""
        response = self._request("GET", "/health")
        return _parse_health_response(response)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request and handle errors."""
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise DocVerifyError(f"Connection failed: {e}") from e
        except httpx.TimeoutException as e:
            raise DocVerifyError(f"Request timed out: {e}") from e

        if response.status_code == 200 or response.status_code == 201:
            return response.json()

        # Map HTTP errors to SDK exceptions
        detail = ""
        try:
            error_data = response.json()
            detail = error_data.get("detail", str(error_data))
        except Exception:
            detail = response.text

        if response.status_code == 401:
            raise AuthenticationError(detail)
        if response.status_code == 403:
            raise AuthenticationError(f"Forbidden: {detail}")
        if response.status_code == 422:
            raise ValidationError(detail)
        if response.status_code == 429:
            raise RateLimitError(detail)

        raise DocVerifyError(detail, status_code=response.status_code)

    def close(self):
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
