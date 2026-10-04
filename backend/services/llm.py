import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from backend.config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Returns the configured LLM. Priority / fallback order:
      1. LLM_PROVIDER env var (gemini | ollama | huggingface)
      2. If the chosen provider fails to init, falls back to huggingface (if HF_TOKEN set)

    Providers:
      gemini      — Google Gemini via AI Studio free tier (GEMINI_API_KEY required)
      ollama      — Local Ollama instance (no API key, GPU optional)
      huggingface — HuggingFace Inference API free serverless tier (HF_TOKEN required)
    """
    provider = settings.llm_provider.lower()

    try:
        return _build_llm(provider)
    except Exception as primary_err:
        # Auto-fallback to HuggingFace if a token is available
        if provider != "huggingface" and settings.hf_token:
            logger.warning(
                "Primary LLM provider '%s' failed (%s). Falling back to HuggingFace.",
                provider,
                primary_err,
            )
            return _build_llm("huggingface")
        raise


def _build_llm(provider: str) -> BaseChatModel:
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

    if provider == "huggingface":
        if not settings.hf_token:
            raise RuntimeError(
                "LLM_PROVIDER=huggingface but HF_TOKEN is not set in .env. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        endpoint = HuggingFaceEndpoint(
            repo_id=settings.hf_model,
            huggingfacehub_api_token=settings.hf_token,
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.01,  # HF endpoint doesn't support exactly 0
            do_sample=False,
        )
        return ChatHuggingFace(llm=endpoint)

    raise RuntimeError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Valid options: 'gemini', 'ollama', 'huggingface'."
    )