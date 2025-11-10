# app/whatsapp.py
import os
import httpx
from typing import Any, Dict

WA_TOKEN = os.getenv("WA_ACCESS_TOKEN", "")
WA_PHONE_ID = os.getenv("WA_PHONE_NUMBER_ID", "")
WA_API_VER = os.getenv("WA_API_VERSION", "v21.0")

def _wa_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}

def _wa_base_url() -> str:
    return f"https://graph.facebook.com/{WA_API_VER}/{WA_PHONE_ID}/messages"

async def send_whatsapp_text(to_number: str, body: str) -> Dict[str, Any]:
    """
    Envía un mensaje de texto por WhatsApp Cloud API.
    to_number: E.164, ej '57300XXXXXXX'
    """
    if not (WA_TOKEN and WA_PHONE_ID):
        return {"ok": False, "error": "Faltan WA_ACCESS_TOKEN o WA_PHONE_NUMBER_ID"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": body[:4096]},
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(_wa_base_url(), headers=_wa_headers(), json=payload)
        data = {}
        try:
            data = r.json() if r.content else {}
        except Exception:
            data = {"raw": r.text}
        return {"ok": 200 <= r.status_code < 300, "status": r.status_code, "data": data}

async def send_typing_on(to_number: str) -> Dict[str, Any]:
    """Marca 'escribiendo...' (states). No todos los entornos lo muestran."""
    if not (WA_TOKEN and WA_PHONE_ID):
        return {"ok": False, "error": "Faltan WA_ACCESS_TOKEN o WA_PHONE_NUMBER_ID"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "reaction",
        "reaction": {"emoji": "✍️"}
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(_wa_base_url(), headers=_wa_headers(), json=payload)
        return {"ok": 200 <= r.status_code < 300, "status": r.status_code}
