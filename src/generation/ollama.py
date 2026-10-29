"""Cliente minimo de la API local de Ollama (sin costo, sin claves)."""

import requests

DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_URL = "http://localhost:11434"


class OllamaUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_URL, num_ctx: int = 8192,
                 timeout: float = 300):
        self.model, self.base_url, self.num_ctx, self.timeout = model, base_url.rstrip("/"), num_ctx, timeout

    def installed_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                "No se pudo conectar con Ollama en %s. Instálalo desde https://ollama.com y ejecuta "
                "`ollama pull %s`." % (self.base_url, self.model)) from exc
        return [m["name"] for m in r.json().get("models", [])]

    def check(self) -> None:
        models = self.installed_models()
        if not any(m == self.model or m.split(":")[0] == self.model for m in models):
            raise OllamaUnavailable(f"Falta el modelo {self.model}: ejecuta `ollama pull {self.model}` "
                                    f"(instalados: {', '.join(models) or 'ninguno'}).")

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        try:
            r = requests.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
                "model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_ctx": self.num_ctx}})
            r.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"Fallo la llamada a Ollama: {exc}") from exc
        return r.json()["message"]["content"].strip()
