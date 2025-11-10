import os, json, asyncio
from typing import Any, Dict
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.settings import get_settings
from app.rag import answer_with_rag
from app.whatsapp import send_whatsapp_text

from app.ingest_qdrant import search_docs

docs = search_docs(text)  # busca los fragmentos relevantes
answer = answer_with_rag(text, docs)

S = get_settings()
app = FastAPI(title="CCP Chatbot")

# --------- estático (opcional) ----------
app.mount("/static", StaticFiles(directory="static", check_dir=False), name="static")

@app.get("/")
def home():
    return {"ok": True, "service": "Chatbot CCP online"}

# ---------- Verificación Webhook (GET) ----------
@app.get("/webhook")
def verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    mode: str | None = Query(None),
    challenge: str | None = Query(None),
    verify_token: str | None = Query(None),
):
    m = hub_mode or mode or ""
    t = hub_verify_token or verify_token or ""
    c = hub_challenge or challenge or ""
    if m == "subscribe" and t == S.WA_VERIFY_TOKEN:
        return PlainTextResponse(c or "", status_code=200)
    return PlainTextResponse("Error de verificación", status_code=403)

# ---------- Util: extraer texto de distintos tipos ----------
def extract_text_from_message(msg: Dict[str, Any]) -> str:
    """
    Soporta:
      - {"type":"text","text":{"body":"..."}}
      - {"type":"interactive","interactive":{"type":"list_reply"/"button_reply", "title"/"id"}}
      - {"type":"button","button":{"text":"..."}}
    """
    mtype = msg.get("type")
    if mtype == "text":
        return (msg.get("text") or {}).get("body", "") or ""
    if mtype == "interactive":
        inter = msg.get("interactive") or {}
        itype = inter.get("type")
        if itype == "list_reply":
            # título visible del ítem seleccionado
            return (inter.get("list_reply") or {}).get("title", "") or ""
        if itype == "button_reply":
            return (inter.get("button_reply") or {}).get("title", "") or ""
    if mtype == "button":
        return (msg.get("button") or {}).get("text", "") or ""
    # Fallback: algunos clientes envían "text" fuera
    return (msg.get("text") or {}).get("body", "") or ""

# ---------- Recepción de mensajes (POST) ----------
@app.post("/webhook")
async def webhook(req: Request):
    try:
        body = await req.json()
        # Logs básicos para depurar en Render (ver en Logs):
        print("Incoming body:", json.dumps(body, ensure_ascii=False))

        value = body.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {})
        statuses = value.get("statuses", [])
        if statuses:
            # acuses de recibo de WhatsApp → ignorar
            return JSONResponse({"ok": True, "status_event": True})

        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"ignored": True})

        msg = messages[0]
        from_number = msg.get("from", "")
        user_text = extract_text_from_message(msg).strip()

        if not user_text:
            await send_whatsapp_text(from_number, "Envíame tu consulta en texto.")
            return JSONResponse({"ok": True, "empty": True})

        # -------- RAG ----------
        try:
            answer = await answer_with_rag(user_text)
        except Exception as e:
            print("RAG error:", str(e))
            answer = "Tu consulta fue recibida, pero hubo un problema temporal al generar la respuesta. Intenta nuevamente."

        # -------- Responder por WhatsApp ----------
        resp = await send_whatsapp_text(from_number, answer)
        print("Send WA resp:", resp)   # ver si Graph devuelve 200/400 en Render Logs

        return JSONResponse({"ok": True})

    except Exception as e:
        print("Webhook error:", str(e))
        # WhatsApp espera 200 siempre para no reintentar en loop
        return JSONResponse({"ok": False, "error": str(e)}, status_code=200)

# ---------- Health ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}
