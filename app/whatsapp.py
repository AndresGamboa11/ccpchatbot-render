# app/whatsapp.py — WhatsApp Cloud API (solo nube, sin override local)
import os
import httpx
import logging
import asyncio
from dotenv import load_dotenv, find_dotenv

# ─────────────────────────────────────────────────────────────
# Carga .env solo si existe y SIN override (para no pisar Render)
# ─────────────────────────────────────────────────────────────
_dotenv = find_dotenv(usecwd=True)
if _dotenv:
    load_dotenv(_dotenv, override=False)

# ─────────────────────────────────────────────────────────────
# Configuración y logging
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger("ccp.whatsapp")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

WA_TOKEN    = (os.getenv("WA_ACCESS_TOKEN") or "").strip()
WA_PHONE_ID = (os.getenv("WA_PHONE_NUMBER_ID") or "").strip()
WA_VER      = (os.getenv("WA_API_VERSION") or "v21.0").strip()

# ─────────────────────────────────────────────────────────────
# Enviar texto
# ─────────────────────────────────────────────────────────────
async def send_whatsapp_text(to_number: str, body: str) -> dict:
    """
    Envía un mensaje de texto al usuario por WhatsApp Cloud API.
    Requiere WA_ACCESS_TOKEN y WA_PHONE_NUMBER_ID configurados en Render.
    """
    if not WA_TOKEN or not WA_PHONE_ID:
        logger.error("❌ Faltan WA_ACCESS_TOKEN o WA_PHONE_NUMBER_ID en el entorno.")
        return {"ok": False, "error": "Falso WA_ACCESS_TOKEN o WA_PHONE_NUMBER_ID"}

    url = f"https://graph.facebook.com/{WA_VER}/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": body[:4096]},
    }

    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            r = await cli.post(url, headers=headers, json=payload)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}
            ok = r.is_success
            if not ok:
                logger.error("❌ Error enviando mensaje: %s", data)
            else:
                logger.info("📤 Mensaje enviado a %s (%s)", to_number, r.status_code)
            return {"ok": ok, "status": r.status_code, "resp": data}
    except Exception as e:
        logger.exception("❌ Excepción enviando mensaje WA: %s", e)
        return {"ok": False, "error": str(e)}

# ─────────────────────────────────────────────────────────────
# Enviar “escribiendo...” (typing)
# ─────────────────────────────────────────────────────────────
# ------------------- Enviar “escribiendo...” -------------------
async def send_typing_on(to_number: str) -> dict:
    """
    Envía la señal 'typing_on' para mostrar que el bot está escribiendo.
    Formato actual de la Cloud API: type='typing', typing='on'
    """
    if not WA_TOKEN or not WA_PHONE_ID:
        logger.warning("⚠️ No se pudo enviar typing_on: faltan credenciales WA.")
        return {"ok": False, "error": "sin credenciales WA"}

    url = f"https://graph.facebook.com/{WA_VER}/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "typing",
        "typing": "on",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as cli:
            r = await cli.post(url, headers=headers, json=payload)
            if r.is_success:
                logger.debug("✏️ typing_on enviado a %s", to_number)
            else:
                logger.debug("typing_on no aceptado: %s", r.text)
            return {"ok": r.is_success, "status": r.status_code}
    except Exception as e:
        logger.debug("No se pudo enviar 'typing_on': %s", e)
        return {"ok": False, "error": str(e)}

