from functools import cache
from os import getenv

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 1024


@cache
def chat() -> ChatOpenAI:
    """Build a ChatOpenAI client from environment variables.

    The instance is cached so repeated calls reuse the same client,
    avoiding redundant construction and enabling HTTP connection reuse.
    """
    api_key = getenv("API_KEY")
    if not api_key:
        raise ValueError("please set your API_KEY in .env file")

    model = getenv("MODEL")
    if not model:
        raise ValueError("please set your MODEL in .env file")

    base_url = getenv("BASE_URL")
    if not base_url:
        raise ValueError("please set your BASE_URL in .env file")

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=DEFAULT_TEMPERATURE,
        max_completion_tokens=DEFAULT_MAX_TOKENS,
    )
