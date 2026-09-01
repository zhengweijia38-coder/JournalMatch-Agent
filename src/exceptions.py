"""Small, user-facing exception hierarchy for the journal RAG application."""


class JournalRAGError(RuntimeError):
    """Base class for expected application failures."""


class ConfigurationError(JournalRAGError):
    """Raised when required application configuration is invalid or missing."""


class PaperProcessingError(JournalRAGError):
    """Raised when a PDF cannot be loaded or analyzed safely."""


class PaperAssessmentError(JournalRAGError):
    """Raised when evidence-based paper quality assessment fails."""


class JournalDatabaseError(JournalRAGError):
    """Raised when the local SQLite journal database is unavailable or invalid."""


class VectorStoreError(JournalRAGError):
    """Raised when the local Chroma index is unavailable or invalid."""


class RetrievalError(JournalRAGError):
    """Raised when journal retrieval cannot produce a valid result."""


class RerankerError(JournalRAGError):
    """Raised when the local cross-encoder cannot rerank candidates."""


class RecommendationError(JournalRAGError):
    """Raised when grounded recommendation generation or validation fails."""
