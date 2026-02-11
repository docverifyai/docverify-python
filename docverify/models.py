"""
DocVerify SDK response models.

Typed dataclasses that mirror the API response schemas.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class EvidenceDetail:
    """Evidence supporting or contradicting a claim."""

    reasoning: str
    source_text: str | None = None
    source_location: dict | None = None


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""

    claim: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "PARTIAL", "AMBIGUOUS"]
    confidence: float
    severity: Literal["critical", "high", "medium", "low"]
    evidence: EvidenceDetail

    @property
    def is_supported(self) -> bool:
        return self.verdict == "SUPPORTED"

    @property
    def is_contradicted(self) -> bool:
        return self.verdict == "CONTRADICTED"

    @property
    def has_issues(self) -> bool:
        return self.verdict in ("CONTRADICTED", "UNSUPPORTED")


@dataclass
class VerificationMetadata:
    """Metadata about the verification process."""

    model_version: str
    processing_time_ms: int
    source_tokens: int
    claims_extracted: int


@dataclass
class VerificationResponse:
    """Complete verification response from the API."""

    verification_id: str
    overall_score: float
    verdict: Literal["ACCURATE", "MINOR_ISSUES", "MAJOR_ISSUES", "UNRELIABLE"]
    claims: list[ClaimVerification]
    metadata: VerificationMetadata

    @property
    def is_accurate(self) -> bool:
        return self.verdict == "ACCURATE"

    @property
    def has_major_issues(self) -> bool:
        return self.verdict in ("MAJOR_ISSUES", "UNRELIABLE")

    @property
    def contradicted_claims(self) -> list[ClaimVerification]:
        return [c for c in self.claims if c.verdict == "CONTRADICTED"]

    @property
    def supported_claims(self) -> list[ClaimVerification]:
        return [c for c in self.claims if c.verdict == "SUPPORTED"]


@dataclass
class HealthResponse:
    """Health check response."""

    status: str
    version: str
    pipeline: str = "unknown"
    database: str = "unknown"
    redis: str = "unknown"


@dataclass
class KeyCreateResponse:
    """Response from API key creation."""

    api_key: str
    key_prefix: str
    tier: str
    rate_limit_per_day: int
    message: str


@dataclass
class FeedbackResponse:
    """Response from submitting feedback."""

    feedback_id: str
    status: str


@dataclass
class FeedbackItem:
    """A single feedback item."""

    feedback_id: str
    verification_id: str
    claim_index: int
    corrected_verdict: str
    notes: str | None
    status: str
    created_at: str


def _parse_verification_response(data: dict) -> VerificationResponse:
    """Parse raw API JSON into typed VerificationResponse."""
    claims = []
    for claim_data in data.get("claims", []):
        evidence_data = claim_data.get("evidence", {})
        evidence = EvidenceDetail(
            reasoning=evidence_data.get("reasoning", ""),
            source_text=evidence_data.get("source_text"),
            source_location=evidence_data.get("source_location"),
        )
        claims.append(
            ClaimVerification(
                claim=claim_data["claim"],
                verdict=claim_data["verdict"],
                confidence=claim_data["confidence"],
                severity=claim_data["severity"],
                evidence=evidence,
            )
        )

    metadata_data = data.get("metadata", {})
    metadata = VerificationMetadata(
        model_version=metadata_data.get("model_version", "unknown"),
        processing_time_ms=metadata_data.get("processing_time_ms", 0),
        source_tokens=metadata_data.get("source_tokens", 0),
        claims_extracted=metadata_data.get("claims_extracted", 0),
    )

    return VerificationResponse(
        verification_id=data["verification_id"],
        overall_score=data["overall_score"],
        verdict=data["verdict"],
        claims=claims,
        metadata=metadata,
    )


def _parse_health_response(data: dict) -> HealthResponse:
    """Parse raw API JSON into typed HealthResponse."""
    return HealthResponse(
        status=data.get("status", "unknown"),
        version=data.get("version", "unknown"),
        pipeline=data.get("pipeline", "unknown"),
        database=data.get("database", "unknown"),
        redis=data.get("redis", "unknown"),
    )
