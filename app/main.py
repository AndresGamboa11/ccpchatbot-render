# app/main.py
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from app.settings import get_settings
from app.whatsapp import send_whatsapp_text
from app.rag import answer_with_rag
from app.mcp_server import router as mcp_router

S = get_settings()
app = FastAPI(title="Chatbot CCP (Render)")
app.include_router(mcp_router)

# CORS básico
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online ✅"}

# ------------ Verificación Webhook (GET) ------------
@app.get("/webhook")
def verify(mode: str = "", hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = ""):
    token = hub_verify_token or ""
    if token == S.TOKEN_VERIFICACION_WA:
        return PlainTextResponse(hub_challenge or "")
    return PlainTextResponse("403", status_code=403)

# ------------ Recepción de mensajes (POST) ----------
@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
        # Navegar el payload típico de WhatsApp Cloud
        entry = body.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"status": "no-message"}, status_code=200)

        msg = messages[0]
        from_number = msg.get("from")
        mtype = msg.get("type")
        text = ""
        if mtype == "text":
            text = msg["text"]["body"]
        elif mtype == "interactive":
            text = msg["interactive"]["list_reply"]["title"] if "list_reply" in msg["interactive"] else ""
        else:
            text = ""

        # Generar respuesta
        reply = await answer_with_rag(text or "")
        await send_whatsapp_text(from_number, reply)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
