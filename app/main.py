# app/main.py
import os, json, logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv, find_dotenv

from app.settings import get_settings
from app.rag import answer_with_rag, debug_qdrant_sample
from app.whatsapp import send_whatsapp_text, send_typing_on
from app.mcp_server import router as mcp_router

# ─────────────────────────────────────────────────────────────
# Cargar .env (NO sobreescribir variables de Render)
# ─────────────────────────────────────────────────────────────
_dotenv = find_dotenv(usecwd=True)
if _dotenv:
    load_dotenv(_dotenv, override=False)

S = get_settings()
app = FastAPI(title="CCP Chatbot")

# ─────────────────────────────────────────────────────────────
# Logs
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ccp")

# ─────────────────────────────────────────────────────────────
# Archivos estáticos
# ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

# ─────────────────────────────────────────────────────────────
# MCP
# ─────────────────────────────────────────────────────────────
app.include_router(mcp_router)

# ─────────────────────────────────────────────────────────────
# Salud y diagnóstico
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}

@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "env": os.environ.get("RENDER", "local"),
        "port": os.environ.get("PORT"),
    }

@app.get("/debug/env")
def debug_env():
    def ok(k): return bool(os.getenv(k))
    return {
        "HF_API_TOKEN": ok("HF_API_TOKEN"),
        "HF_EMBED_MODEL": os.getenv("HF_EMBED_MODEL"),
        "QDRANT_URL": ok("QDRANT_URL"),
        "QDRANT_API_KEY": ok("QDRANT_API_KEY"),
        "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION"),
        "GROQ_API_KEY": ok("GROQ_API_KEY"),
        "GROQ_MODEL": os.getenv("GROQ_MODEL"),
        "WA_ACCESS_TOKEN": ok("WA_ACCESS_TOKEN"),
        "WA_PHONE_NUMBER_ID": ok("WA_PHONE_NUMBER_ID"),
        "WA_VERIFY_TOKEN": ok("WA_VERIFY_TOKEN"),
        "WA_API_VERSION": os.getenv("WA_API_VERSION"),
    }

@app.get("/debug/rag")
def debug_rag(q: str = ""):
    if not q.strip():
        return JSONResponse({"error": "falta parámetro q"}, status_code=400)
    try:
        ans = answer_with_rag(q)
        return {"query": q, "answer": ans}
    except Exception as e:
        logger.exception("[/debug/rag] Error")
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────────────────────
# Debug Qdrant
# ─────────────────────────────────────────────────────────────
@app.get("/debug/qdrant")
def debug_qdrant():
    try:
        info = debug_qdrant_sample()
        return info
    except Exception as e:
        logger.exception("[/debug/qdrant] Error")
        return JSONResponse({"error": str(e)}, status_code=500)

# ─────────────────────────────────────────────────────────────
# WhatsApp Webhook
# ─────────────────────────────────────────────────────────────
WA_VERIFY_TOKEN = (os.getenv("WA_VERIFY_TOKEN") or "").strip()

# 1) Verificación webhook (GET)
@app.get("/webhook")
async def verify_webhook(
    mode: str = "",
    hub_mode: str = "",
    hub_challenge: str = "",
    hub_verify_token: str = "",
):
    _mode = (mode or hub_mode or "").lower()
    _token = (hub_verify_token or "").strip()

    if _mode == "subscribe" and WA_VERIFY_TOKEN and _token == WA_VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "OK", status_code=200)

    return PlainTextResponse("Forbidden", status_code=403)


# 2) Extraer mensaje entrante
def _extract_wa_message(payload: dict):
    try:
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        msgs = value.get("messages") or []
        if msgs:
            msg = msgs[0]
            from_number = msg.get("from")
            msg_type = msg.get("type")
            text = None

            if msg_type == "text":
                text = (msg.get("text", {}).get("body") or "").strip()

            elif msg_type == "interactive":
                interactive = msg.get("interactive", {})
                itype = interactive.get("type")
                if itype == "button_reply":
                    text = interactive.get("button_reply", {}).get("title", "").strip()
                elif itype == "list_reply":
                    text = interactive.get("list_reply", {}).get("title", "").strip()

            elif msg_type == "image":
                text = "imagen"

            return from_number, text

        if "statuses" in value:
            return None, None

    except Exception:
        return None, None

    return None, None


# 3) Recepción del webhook (POST)
@app.post("/webhook")
async def receive_webhook(request: Request): ##
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": True}, status_code=200)

    logger.info("WA IN: %s", json.dumps(payload)[:2000])

    from_number, text = _extract_wa_message(payload)
    if not from_number:
        return JSONResponse({"ok": True, "note": "no user message"}, status_code=200)

    try:
        await send_typing_on(from_number)
    except Exception:
        pass

    if not text:
        await send_whatsapp_text(from_number, "Hola 👋, por favor escribe tu consulta.")
        return JSONResponse({"ok": True}, status_code=200)

    # ----- RAG -----
    try:
        answer = answer_with_rag(text)
        if not answer.strip():
            answer = (
                "Lo siento, no encontré información exacta sobre eso. "
                "¿Puedes reformular tu pregunta?"
            )
    except Exception:
        logger.exception("RAG error")
        answer = "Ocurrió un error interno, intenta nuevamente."

    # ----- enviar respuesta -----
    try:
        wa_res = await send_whatsapp_text(from_number, answer)
        logger.info("WA OUT: %s", wa_res)
    except Exception:
        logger.exception("WA send error")

    return JSONResponse({"ok": True}, status_code=200)
