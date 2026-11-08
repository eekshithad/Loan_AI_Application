"""
Ollama LLM Client
Wraps the Ollama Python SDK for structured calls to llama3.1.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    """Thin wrapper around the Ollama Python library."""

    def __init__(self, model: str = _DEFAULT_MODEL, base_url: str = _DEFAULT_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._client = None
        self._available = None

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.base_url)
        return self._client

    def is_available(self) -> tuple[bool, str]:
        """Check if Ollama is running and the model is available."""
        if self._available is not None:
            return self._available

        try:
            client = self._get_client()
            list_response = client.list()
            # ollama 0.6+ returns ListResponse with .models list of Model objects
            model_names = [m.model for m in list_response.models]
            # Check if our model is listed (allow partial match like "llama3.1")
            found = any(self.model in m or m.startswith(self.model.split(":")[0])
                        for m in model_names)
            if found:
                self._available = (True, f"Ollama running, model '{self.model}' available.")
            else:
                self._available = (
                    False,
                    f"Ollama running but model '{self.model}' not found. "
                    f"Available: {model_names}"
                )
        except Exception as exc:
            self._available = (False, f"Ollama unavailable: {exc}")

        return self._available

    def chat(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """
        Send a chat prompt and return the response text.
        Raises RuntimeError if Ollama is unavailable.
        """
        ok, msg = self.is_available()
        if not ok:
            raise RuntimeError(f"Ollama not available: {msg}")

        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
            )
            return response.message.content
        except Exception as exc:
            logger.error(f"Ollama chat error: {exc}")
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    def extract_json(self, prompt: str, system: str = "") -> dict:
        """
        Call LLM and attempt to parse JSON from the response.
        Returns {} on failure.
        """
        import json, re
        try:
            raw = self.chat(prompt, system=system)
            # Find JSON block in response
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            # Try parsing full response
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"JSON extraction from LLM failed: {exc}")
            return {}

