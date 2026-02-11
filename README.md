# DocVerify Python SDK

Official Python client for the [DocVerify](https://docverify.dev) document verification API. Verify AI-generated outputs against source documents with claim-level accuracy reports.

## Installation

```bash
pip install docverify
```

## Quick Start

```python
from docverify import DocVerifyClient

client = DocVerifyClient(api_key="dv_your_api_key")

result = client.verify(
    source="The agreement terminates after 60 days written notice.",
    ai_output="The termination clause requires 90 days notice.",
    domain="legal",
)

print(f"Score: {result.overall_score}")  # 0.42
print(f"Verdict: {result.verdict}")      # MAJOR_ISSUES

for claim in result.claims:
    print(f"  {claim.verdict}: {claim.claim}")
    if claim.is_contradicted:
        print(f"    Evidence: {claim.evidence.source_text}")
```

## Core Methods

### Verify text

```python
result = client.verify(
    source="Source document text...",
    ai_output="AI-generated summary...",
    source_type="text",        # text, pdf, docx
    task_type="extraction",    # extraction, summary, analysis
    domain="general",          # general, legal, financial, medical
    return_evidence=True,
    severity_threshold="low",  # low, medium, high
)
```

### Verify a file (PDF, DOCX, TXT)

```python
result = client.verify_file(
    file_path="contract.pdf",
    ai_output="AI-extracted contract terms...",
    domain="legal",
)
```

### Batch verification (multiple outputs, one source)

```python
results = client.batch_verify(
    source="Source document...",
    ai_outputs=[
        {"id": "model_a", "text": "Output from model A"},
        {"id": "model_b", "text": "Output from model B"},
    ],
)
```

### Compare outputs

```python
comparison = client.compare(
    source="Source document...",
    ai_outputs=[
        {"id": "v1", "text": "First version"},
        {"id": "v2", "text": "Second version"},
    ],
)
print(comparison["summary"])  # {agreed: 5, disagreed: 1}
```

## Response Objects

### VerificationResponse

| Field | Type | Description |
|-------|------|-------------|
| `verification_id` | `str` | Unique ID for this verification |
| `overall_score` | `float` | 0.0 (all wrong) to 1.0 (all correct) |
| `verdict` | `str` | ACCURATE, MINOR_ISSUES, MAJOR_ISSUES, UNRELIABLE |
| `claims` | `list[ClaimVerification]` | Per-claim results |
| `metadata` | `VerificationMetadata` | Processing info |

Helper properties: `is_accurate`, `has_major_issues`, `contradicted_claims`, `supported_claims`

### ClaimVerification

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The extracted claim text |
| `verdict` | `str` | SUPPORTED, CONTRADICTED, UNSUPPORTED, PARTIAL, AMBIGUOUS |
| `confidence` | `float` | Model confidence (0-1) |
| `severity` | `str` | critical, high, medium, low |
| `evidence` | `EvidenceDetail` | Source evidence |

Helper properties: `is_supported`, `is_contradicted`, `has_issues`

## Error Handling

```python
from docverify import DocVerifyClient, AuthenticationError, RateLimitError, DocVerifyError

client = DocVerifyClient(api_key="dv_your_key")

try:
    result = client.verify(source="...", ai_output="...")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, try again later")
except DocVerifyError as e:
    print(f"API error ({e.status_code}): {e.message}")
```

## Configuration

```python
client = DocVerifyClient(
    api_key="dv_your_key",
    base_url="https://api.docverify.dev",  # custom API endpoint
    timeout=120.0,                          # request timeout in seconds
)
```

The client can be used as a context manager:

```python
with DocVerifyClient(api_key="dv_your_key") as client:
    result = client.verify(source="...", ai_output="...")
```

## Requirements

- Python >= 3.10
- httpx >= 0.25.0

## License

MIT
