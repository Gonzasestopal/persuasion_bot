from dataclasses import dataclass


@dataclass
class DomainError(Exception):
    """Base class for all domain-level errors.

    These represent business or validation failures that occur within
    the application's core logic, independent of transport concerns.
    Raise subclasses of this in services, repositories, or adapters
    when the problem is related to business rules or domain state.
    """

    message: str
    code: str = 'domain_error'


class ConfigError(DomainError):
    """Raised when the system is misconfigured.

    Use this for missing or invalid environment variables, API keys,
    or other critical configuration values that prevent the app from
    starting or operating correctly.
    """

    code = 'missing or misconfigured setting'


# 4xx
class InvalidStartMessage(DomainError):  # 422
    code = 'invalid_start_message'


class InvalidContinuationMessage(DomainError):  # 422
    code = 'invalid_continuation_message'


class ConversationNotFound(DomainError):  # 404
    code = 'conversation_not_found'


class ConversationExpired(DomainError):  # 404
    code = 'conversation_expired'


class LLMTimeout(DomainError):  # 503
    code = 'llm_timeout'


class LLMServiceError(DomainError):  # 502 or 503 (choose)
    code = 'llm_service_error'


class InvalidTopic(DomainError):  # 422:
    code = 'invalid_topic'
