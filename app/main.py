# app/main.py
import os, json, asyncio, logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.settings import get_settings
from app.rag import answer_with_rag
from app.whatsapp import send_whatsapp_text, send_typing_on
from app.mcp_server import router as mcp_router

S = get_settings()
app = FastAPI(title="CCP Chatbot")

# Logs útiles
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ccp")

# estático opcional
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# incluye MCP
app.include_router(mcp_router)

@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}

@app.get("/healthz")
async def healthz():
    return {"ok": True, "env": "render", "port": os.environ.get("PORT")}

# -------------------- WhatsApp Webhook --------------------

WA_VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN", "")

# 1) Verificación de webhook (GET)
@app.get("/webhook")
async def verify_webhook(mode: str = "", hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = ""):
    # Meta puede enviar 'mode' ó 'hub.mode', etc. Capturo los dos por compatibilidad.
    _mode = (mode or hub_mode or "").lower()
    _token = (hub_verify_token or "").strip()

    if _mode == "subscribe" and WA_VERIFY_TOKEN and _token == WA_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "OK", status_code=200)
    return PlainTextResponse("Forbidden", status_code=403)

def _extract_wa_message(payload: dict):
    """
    Devuelve (from_number, text) si hay un mensaje de usuario.
    Si no hay mensaje (p.ej. es 'statuses'), devuelve (None, None).
    """
    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        # mensajes entrantes
        if "messages" in value and value.get("messages"):
            msg = value["messages"][0]
            from_number = msg.get("from")
            text = None
            t = msg.get("type")
            if t == "text":
                text = (msg["text"].get("body") or "").strip()
            elif t == "interactive":
                # botones/listas
                interactive = msg.get("interactive", {})
                if interactive.get("type") == "button_reply":
                    text = (interactive["button_reply"].get("title") or "").strip()
                elif interactive.get("type") == "list_reply":
                    text = (interactive["list_reply"].get("title") or "").strip()
            elif t == "image":
                text = "imagen"  # podrías manejar OCR si lo deseas
            else:
                text = None
            return from_number, text

        # estatus de entrega
        if "statuses" in value:
            return None, None
    except Exception:
        return None, None

    return None, None

# 2) Recepción de eventos (POST)
@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": True}, status_code=200)

    logger.info("WA IN: %s", json.dumps(payload)[:2000])

    from_number, text = _extract_wa_message(payload)
    if not from_number:
        # Importante: siempre responder 200 para que Meta no reintente infinitamente
        return JSONResponse({"ok": True, "note": "no user message"}, status_code=200)

    # Señal de 'escribiendo' (opcional)
    try:
        await send_typing_on(from_number)
    except Exception:
        pass

    # Si no hay texto, responder ayuda mínima
    if not text:
        await send_whatsapp_text(from_number, "Hola 👋, por favor escribe tu consulta en texto.")
        return JSONResponse({"ok": True}, status_code=200)

    # Llama al RAG y responde
    try:
        answer = await answer_with_rag(text)
        if not answer or not answer.strip():
            answer = "Lo siento, no encontré información exacta sobre eso. ¿Puedes reformular tu pregunta?"
    except Exception as e:
        logger.exception("RAG error")
        answer = f"Hubo un error procesando tu solicitud. Intenta de nuevo. (detalle: {str(e)[:180]})"

    wa_res = await send_whatsapp_text(from_number, answer)
    logger.info("WA OUT: %s", wa_res)

    return JSONResponse({"ok": True}, status_code=200)
