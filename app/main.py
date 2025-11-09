import os, json, asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.settings import get_settings
from app.rag import answer_with_rag
from app.whatsapp import send_whatsapp_text

S = get_settings()
app = FastAPI(title="CCP Chatbot")

# estático opcional
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}

# -------- Verificación del webhook (GET) -------------
@app.get("/webhook")
def verify(mode: str = "", hub_mode: str = "", hub_challenge: str = "", hub_verify_token: str = "",
          hub_challenge_alt: str = "", hub_mode_alt: str = "", hub_verify_token_alt: str = ""):
    # Soportar variantes
    token = hub_verify_token or hub_verify_token_alt or hub_verify_token or ""
    challenge = hub_challenge or hub_challenge_alt or ""
    if (mode == "subscribe" or hub_mode in ("subscribe", "subscribe")) and token == S.WA_VERIFY_TOKEN:
        return PlainTextResponse(challenge, status_code=200)
    return PlainTextResponse("Error de verificación", status_code=403)

# -------- Recepción de mensajes (POST) ---------------
@app.post("/webhook")
async def webhook(req: Request):
    body = await req.json()
    try:
        entry = body["entry"][0]["changes"][0]["value"]
        msgs = entry.get("messages", [])
        if not msgs:
            return JSONResponse({"ignored": True})

        msg = msgs[0]
        from_number = msg["from"]
        text = (msg.get("text") or {}).get("body", "").strip()

        if not text:
            await send_whatsapp_text(from_number, "Envíame un texto con tu consulta.")
            return JSONResponse({"ok": True})

        answer = await answer_with_rag(text)
        await send_whatsapp_text(from_number, answer)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

# -------- Health ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
-----------------------------
