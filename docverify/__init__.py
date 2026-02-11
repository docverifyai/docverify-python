"""
DocVerify Python SDK.

Usage:
    from docverify import DocVerifyClient

    client = DocVerifyClient(api_key="dv_...")
    result = client.verify(
        source="The contract term is 24 months.",
        ai_output="The contract term is 36 months.",
        domain="legal",
    )
    print(result.verdict)  # "MAJOR_ISSUES"
    for claim in result.claims:
        print(f"  {claim.verdict}: {claim.claim}")
"""

from docverify.client import DocVerifyClient
from docverify.errors import AuthenticationError, DocVerifyError, RateLimitError, ValidationError
from docverify.models import (
    ClaimVerification,
    EvidenceDetail,
    HealthResponse,
    VerificationMetadata,
    VerificationResponse,
)

__all__ = [
    "DocVerifyClient",
    "DocVerifyError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "VerificationResponse",
    "ClaimVerification",
    "EvidenceDetail",
    "VerificationMetadata",
    "HealthResponse",
]

__version__ = "1.0.0"
