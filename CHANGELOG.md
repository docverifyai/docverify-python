# Changelog

## 1.0.0 (2026-02-11)

### Added
- Initial release
- `DocVerifyClient` with full API coverage
- Core verification: `verify()`, `verify_file()`, `verify_stream()`
- Batch operations: `batch_verify()`, `compare()`
- Enterprise features: teams, policies, alerts, workflows
- Typed response models: `VerificationResponse`, `ClaimVerification`, `EvidenceDetail`
- Error classes: `AuthenticationError`, `RateLimitError`, `ValidationError`
- Context manager support
- PEP 561 type stub marker (`py.typed`)
