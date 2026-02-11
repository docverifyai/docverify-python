"""
DocVerify SDK error types.

All errors inherit from DocVerifyError for easy catch-all handling.
"""


class DocVerifyError(Exception):
    """Base exception for all DocVerify SDK errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(DocVerifyError):
    """Raised when the API key is invalid or missing."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, status_code=401)


class RateLimitError(DocVerifyError):
    """Raised when the rate limit has been exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


class ValidationError(DocVerifyError):
    """Raised when the request fails validation."""

    def __init__(self, message: str = "Request validation failed"):
        super().__init__(message, status_code=422)
