import logging

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.mock_provider import MockProvider
from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


def build_provider(settings: Settings, embeddings: EmbeddingService | None = None) -> LLMProvider:
    """Real LLM via LangChain when a key is configured, otherwise the extractive mock.

    To add another vendor, build its LangChain chat model here and wrap it in
    ``LangChainProvider`` (e.g. ``ChatOpenAI``) - nothing else in the app changes.
    """
    choice = settings.llm_provider.lower()
    if choice in ("auto", "anthropic") and settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic

        from app.llm.langchain_provider import LangChainProvider

        logger.info("LLM provider: anthropic via LangChain (%s)", settings.anthropic_model)
        model = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
            max_retries=2,
        )
        return LangChainProvider(model, name="anthropic")
    if choice == "anthropic":
        logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set; using mock")
    logger.info("LLM provider: mock (extractive)")
    return MockProvider(embeddings)
