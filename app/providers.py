# app/providers.py
import os, httpx, json, logging

log = logging.getLogger("groq")
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "gemma2-9b-it")

class GroqClient:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str = None, model: str = None, timeout: int = 60):
        self.api_key = (api_key or GROQ_API_KEY).strip()
        self.model   = (model or GROQ_MODEL).strip()
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY vacío o faltante")

    def chat(self, messages, max_tokens: int = 400, temperature: float = 0.2, stream: bool = False):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
        }

        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(self.URL, headers=headers, json=payload)
            if r.status_code >= 400:
                # Log detallado para ver exactamente por qué 400
                log.error("Groq 4xx/5xx: %s\nReq: %s\nResp: %s",
                          r.status_code, json.dumps(payload, ensure_ascii=False),
                          r.text)
                # Lanza una excepción con el cuerpo (útil para tu middleware)
                try:
                    detail = r.json()
                except Exception:
                    detail = {"error": r.text}
                raise RuntimeError(f"Groq error {r.status_code}: {detail}")

            data = r.json()
            # si no haces streaming:
            return data["choices"][0]["message"]["content"]
