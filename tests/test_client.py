"""
Tests for the DocVerify Python SDK client.

Sprint 21, Story 21.1.1, Tasks T-325 through T-329.

Uses respx to mock httpx requests — no real server needed.
"""

import json
import tempfile
from pathlib import Path

import httpx
import pytest
import respx

from docverify import DocVerifyClient
from docverify.errors import AuthenticationError, DocVerifyError, RateLimitError, ValidationError
from docverify.models import ClaimVerification, VerificationMetadata, VerificationResponse

BASE_URL = "https://api.docverify.dev"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_VERIFY_RESPONSE = {
    "verification_id": "ver_abc123",
    "overall_score": 0.42,
    "verdict": "MAJOR_ISSUES",
    "claims": [
        {
            "claim": "The contract requires 90 days notice",
            "verdict": "CONTRADICTED",
            "confidence": 0.94,
            "severity": "critical",
            "evidence": {
                "source_text": "terminates after 60 days written notice",
                "source_location": {"page": 3, "paragraph": 2},
                "reasoning": "Source specifies 60 days, not 90 days",
            },
        },
        {
            "claim": "The contract was signed in 2024",
            "verdict": "SUPPORTED",
            "confidence": 0.88,
            "severity": "low",
            "evidence": {
                "source_text": "Signed January 15, 2024",
                "source_location": {"page": 1, "paragraph": 1},
                "reasoning": "Date matches source document",
            },
        },
    ],
    "metadata": {
        "model_version": "docverify-nli-v1.0",
        "processing_time_ms": 340,
        "source_tokens": 4200,
        "claims_extracted": 2,
    },
}

SAMPLE_BATCH_RESPONSE = {
    "results": [
        {"verification_id": "ver_batch_1", "verdict": "ACCURATE", "overall_score": 0.95},
        {"verification_id": "ver_batch_2", "verdict": "MAJOR_ISSUES", "overall_score": 0.3},
    ]
}

SAMPLE_COMPARE_RESPONSE = {
    "results": [
        {"verification_id": "ver_cmp_1", "verdict": "ACCURATE"},
        {"verification_id": "ver_cmp_2", "verdict": "MAJOR_ISSUES"},
    ],
    "comparison": [
        {
            "claim": "The price is $100",
            "agreement": "disagreed",
            "verdicts": {"output_1": "SUPPORTED", "output_2": "CONTRADICTED"},
        }
    ],
    "summary": {"total_claims": 3, "agreed": 2, "disagreed": 1},
}

SAMPLE_HEALTH_RESPONSE = {
    "status": "ok",
    "version": "0.1.0",
    "pipeline": "loaded",
    "database": "connected",
    "redis": "connected",
}


# ===========================================================================
# Auth tests (T-327)
# ===========================================================================


class TestAuth:
    """Test authentication behavior."""

    def test_constructor_requires_api_key_prefix(self):
        """API key must start with 'dv_'."""
        with pytest.raises(ValueError, match="must start with 'dv_'"):
            DocVerifyClient(api_key="bad_key")

    def test_constructor_accepts_valid_key(self):
        """Valid API key is accepted."""
        client = DocVerifyClient(api_key="dv_test123")
        assert client.api_key == "dv_test123"
        client.close()

    def test_custom_base_url(self):
        """Custom base URL is set correctly."""
        client = DocVerifyClient(api_key="dv_test", base_url="https://custom.example.com/")
        assert client.base_url == "https://custom.example.com"
        client.close()

    @respx.mock
    def test_bearer_token_sent(self):
        """API key is sent as Bearer token in Authorization header."""
        route = respx.get(f"{BASE_URL}/health").mock(
            return_value=httpx.Response(200, json=SAMPLE_HEALTH_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_testkey999")
        client.get_health()
        client.close()

        assert route.called
        request = route.calls[0].request
        assert request.headers["authorization"] == "Bearer dv_testkey999"

    @respx.mock
    def test_user_agent_header(self):
        """User-Agent header is set to docverify-python/version."""
        route = respx.get(f"{BASE_URL}/health").mock(
            return_value=httpx.Response(200, json=SAMPLE_HEALTH_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        client.get_health()
        client.close()

        request = route.calls[0].request
        assert "docverify-python" in request.headers["user-agent"]

    def test_context_manager(self):
        """Client works as a context manager."""
        with DocVerifyClient(api_key="dv_ctx") as client:
            assert client.api_key == "dv_ctx"


# ===========================================================================
# Core method tests (T-326)
# ===========================================================================


class TestVerify:
    """Test verify() method."""

    @respx.mock
    def test_verify_returns_typed_response(self):
        """verify() returns a VerificationResponse with typed fields."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERIFY_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        result = client.verify(source="some source text", ai_output="some AI output")
        client.close()

        assert isinstance(result, VerificationResponse)
        assert result.verification_id == "ver_abc123"
        assert result.overall_score == 0.42
        assert result.verdict == "MAJOR_ISSUES"
        assert result.has_major_issues is True
        assert result.is_accurate is False

    @respx.mock
    def test_verify_parses_claims(self):
        """verify() correctly parses claim data."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERIFY_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        result = client.verify(source="src", ai_output="out")
        client.close()

        assert len(result.claims) == 2
        contradicted = result.claims[0]
        assert isinstance(contradicted, ClaimVerification)
        assert contradicted.verdict == "CONTRADICTED"
        assert contradicted.confidence == 0.94
        assert contradicted.is_contradicted is True
        assert contradicted.has_issues is True
        assert contradicted.evidence.source_text == "terminates after 60 days written notice"

        supported = result.claims[1]
        assert supported.is_supported is True
        assert supported.has_issues is False

    @respx.mock
    def test_verify_parses_metadata(self):
        """verify() correctly parses metadata."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERIFY_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        result = client.verify(source="src", ai_output="out")
        client.close()

        assert isinstance(result.metadata, VerificationMetadata)
        assert result.metadata.processing_time_ms == 340
        assert result.metadata.claims_extracted == 2
        assert result.metadata.model_version == "docverify-nli-v1.0"

    @respx.mock
    def test_verify_sends_correct_payload(self):
        """verify() sends all parameters in the request body."""
        route = respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERIFY_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        client.verify(
            source="src doc",
            ai_output="ai text",
            domain="legal",
            task_type="summary",
            severity_threshold="high",
            return_evidence=False,
        )
        client.close()

        body = json.loads(route.calls[0].request.content)
        assert body["source"] == "src doc"
        assert body["ai_output"] == "ai text"
        assert body["domain"] == "legal"
        assert body["task_type"] == "summary"
        assert body["options"]["severity_threshold"] == "high"
        assert body["options"]["return_evidence"] is False

    @respx.mock
    def test_verify_helper_properties(self):
        """VerificationResponse helper properties work."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERIFY_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        result = client.verify(source="src", ai_output="out")
        client.close()

        assert len(result.contradicted_claims) == 1
        assert len(result.supported_claims) == 1
        assert result.contradicted_claims[0].claim == "The contract requires 90 days notice"


class TestVerifyFile:
    """Test verify_file() method."""

    @respx.mock
    def test_verify_file_sends_multipart(self):
        """verify_file() sends multipart form data."""
        route = respx.post(f"{BASE_URL}/v1/verify/file").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERIFY_RESPONSE)
        )

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is a test document.")
            tmp_path = f.name

        try:
            client = DocVerifyClient(api_key="dv_test")
            result = client.verify_file(file_path=tmp_path, ai_output="AI output here")
            client.close()

            assert isinstance(result, VerificationResponse)
            assert route.called
        finally:
            Path(tmp_path).unlink()

    def test_verify_file_missing_file_raises(self):
        """verify_file() raises FileNotFoundError for missing files."""
        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(FileNotFoundError):
            client.verify_file(file_path="/nonexistent/file.pdf", ai_output="test")
        client.close()

    def test_verify_file_unsupported_type_raises(self):
        """verify_file() raises ValueError for unsupported file types."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            tmp_path = f.name

        try:
            client = DocVerifyClient(api_key="dv_test")
            with pytest.raises(ValueError, match="Unsupported file type"):
                client.verify_file(file_path=tmp_path, ai_output="test")
            client.close()
        finally:
            Path(tmp_path).unlink()


class TestBatchVerify:
    """Test batch_verify() method."""

    @respx.mock
    def test_batch_verify_returns_results(self):
        """batch_verify() returns a list of result dicts."""
        route = respx.post(f"{BASE_URL}/v1/verify/batch").mock(
            return_value=httpx.Response(200, json=SAMPLE_BATCH_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        results = client.batch_verify(
            source="source text",
            ai_outputs=[
                {"id": "out1", "text": "first output"},
                {"id": "out2", "text": "second output"},
            ],
        )
        client.close()

        assert len(results) == 2
        assert results[0]["verdict"] == "ACCURATE"
        assert results[1]["verdict"] == "MAJOR_ISSUES"

        body = json.loads(route.calls[0].request.content)
        assert len(body["ai_outputs"]) == 2


class TestCompare:
    """Test compare() method."""

    @respx.mock
    def test_compare_returns_comparison(self):
        """compare() returns results, comparison, and summary."""
        respx.post(f"{BASE_URL}/v1/verify/compare").mock(
            return_value=httpx.Response(200, json=SAMPLE_COMPARE_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        result = client.compare(
            source="source text",
            ai_outputs=[
                {"id": "out1", "text": "first output"},
                {"id": "out2", "text": "second output"},
            ],
        )
        client.close()

        assert "results" in result
        assert "comparison" in result
        assert "summary" in result
        assert result["summary"]["agreed"] == 2
        assert result["summary"]["disagreed"] == 1


class TestHealth:
    """Test get_health() method."""

    @respx.mock
    def test_get_health(self):
        """get_health() returns typed HealthResponse."""
        respx.get(f"{BASE_URL}/health").mock(
            return_value=httpx.Response(200, json=SAMPLE_HEALTH_RESPONSE)
        )

        client = DocVerifyClient(api_key="dv_test")
        health = client.get_health()
        client.close()

        assert health.status == "ok"
        assert health.version == "0.1.0"
        assert health.pipeline == "loaded"


# ===========================================================================
# Error handling tests (T-328)
# ===========================================================================


class TestErrorHandling:
    """Test error mapping from HTTP status codes to SDK exceptions."""

    @respx.mock
    def test_401_raises_authentication_error(self):
        """401 response maps to AuthenticationError."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )

        client = DocVerifyClient(api_key="dv_badkey")
        with pytest.raises(AuthenticationError) as exc_info:
            client.verify(source="src", ai_output="out")
        client.close()

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.message

    @respx.mock
    def test_403_raises_authentication_error(self):
        """403 response maps to AuthenticationError with Forbidden prefix."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(403, json={"detail": "IP not allowed"})
        )

        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(AuthenticationError) as exc_info:
            client.verify(source="src", ai_output="out")
        client.close()

        assert "Forbidden" in exc_info.value.message

    @respx.mock
    def test_422_raises_validation_error(self):
        """422 response maps to ValidationError."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(422, json={"detail": "Missing required field"})
        )

        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(ValidationError) as exc_info:
            client.verify(source="src", ai_output="out")
        client.close()

        assert exc_info.value.status_code == 422

    @respx.mock
    def test_429_raises_rate_limit_error(self):
        """429 response maps to RateLimitError."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(429, json={"detail": "Rate limit exceeded"})
        )

        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(RateLimitError) as exc_info:
            client.verify(source="src", ai_output="out")
        client.close()

        assert exc_info.value.status_code == 429

    @respx.mock
    def test_500_raises_docverify_error(self):
        """500 response maps to base DocVerifyError with status code."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(500, json={"detail": "Internal server error"})
        )

        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(DocVerifyError) as exc_info:
            client.verify(source="src", ai_output="out")
        client.close()

        assert exc_info.value.status_code == 500
        assert "Internal server error" in exc_info.value.message

    @respx.mock
    def test_non_json_error_response(self):
        """Non-JSON error response is handled gracefully."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            return_value=httpx.Response(502, text="Bad Gateway")
        )

        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(DocVerifyError) as exc_info:
            client.verify(source="src", ai_output="out")
        client.close()

        assert exc_info.value.status_code == 502
        assert "Bad Gateway" in exc_info.value.message

    @respx.mock
    def test_connection_error(self):
        """Connection failure maps to DocVerifyError."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(DocVerifyError, match="Connection failed"):
            client.verify(source="src", ai_output="out")
        client.close()

    @respx.mock
    def test_timeout_error(self):
        """Timeout maps to DocVerifyError."""
        respx.post(f"{BASE_URL}/v1/verify").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )

        client = DocVerifyClient(api_key="dv_test")
        with pytest.raises(DocVerifyError, match="Request timed out"):
            client.verify(source="src", ai_output="out")
        client.close()
