"""Small adapter around Google's maintained ``google-genai`` SDK.

The retrieval code deliberately consumes the legacy-shaped
``{"embedding": ...}`` dictionary so it stays independent from any API SDK.
This module is the only place that translates the current SDK responses.
"""
import os

from google import genai
from google.genai import types

from prompt import PROMPT_WORKAW


DEFAULT_GENERATION_MODEL = "gemini-3.8-flash"


def create_client(api_key):
    """Create one reusable client with a finite network timeout."""
    if not api_key or not str(api_key).strip():
        return None
    return genai.Client(
        api_key=str(api_key).strip(),
        http_options=types.HttpOptions(
            timeout=45_000,
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1,
                max_delay=4,
                exp_base=2,
                jitter=1,
            ),
        ),
    )


def generation_model():
    """Allow a deployment to change models without editing source code."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GENERATION_MODEL).strip() or DEFAULT_GENERATION_MODEL


def generation_config():
    return types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        max_output_tokens=1024,
        response_mime_type="text/plain",
        system_instruction=PROMPT_WORKAW,
        safety_settings=[
            types.SafetySetting(category=category, threshold="BLOCK_NONE")
            for category in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ],
    )


def create_chat(client, history):
    if client is None:
        raise ValueError("Gemini client is not configured")
    return client.chats.create(
        model=generation_model(),
        config=generation_config(),
        history=history,
    )


def embedding_function(client):
    """Return a callable compatible with the SDK-independent RAG functions."""
    if client is None:
        return None

    def embed_content(*, model, content, task_type=None, request_options=None):
        del request_options  # Timeout/retry policy is configured on the client.
        is_batch = isinstance(content, (list, tuple))
        config = None
        if task_type:
            config = types.EmbedContentConfig(task_type=str(task_type).upper())
        response = client.models.embed_content(
            model=model,
            contents=list(content) if isinstance(content, tuple) else content,
            config=config,
        )
        embeddings = response.embeddings or []
        values = [list(item.values or []) for item in embeddings]
        if is_batch:
            if len(values) != len(content):
                raise ValueError("Gemini returned an unexpected embedding count")
            return {"embedding": values}
        if len(values) != 1:
            raise ValueError("Gemini returned an unexpected embedding count")
        return {"embedding": values[0]}

    return embed_content
