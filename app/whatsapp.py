# app/whatsapp.py
import httpx
from app.settings import get_settings

S = get_settings()

async def send_whatsapp_text(to_number: str, body: str) -> dict:
    base = f"https://graph.facebook.com/{S.VERSION_WA}/{S.ID_NUMERO_WA}/messages"
    headers = {
        "Authorization": f"Bearer {S.TOKEN_WA}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body},
    }
    async with httpx.AsyncClient(timeout=S.TIMEOUT_HTTP) as client:
        r = await client.post(base, headers=headers, json=payload)
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}
