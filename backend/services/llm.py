from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from backend.config.settings import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set in .env")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            max_retries=2,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    raise RuntimeError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'ollama'.")