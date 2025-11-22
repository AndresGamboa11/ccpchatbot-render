# app/mcp_server.py
"""
Servidor MCP ligero expuesto por HTTP.

Rutas (una vez montado con prefix="/mcp" en main.py):
- GET  /mcp              → info básica MCP
- GET  /mcp/health       → estado del servicio
- POST /mcp/ping         → echo de prueba
- POST /mcp/ask          → consulta al RAG (y opcionalmente envía por WhatsApp)
"""

from typing import Any, Dict
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import os
import platform
import datetime as dt
import logging

from app.rag import answer_with_rag

# settings (opcional, por si quieres usarlas después)
try:
    from app.settings import get_settings
    S = get_settings()
except Exception:
    S = None

log = logging.getLogger("mcp")

# ---------------------------------------------------------------------
# Fallback WhatsApp: si no se puede importar tu cliente, usamos HTTP
# ---------------------------------------------------------------------
try:
    from app.whatsapp import send_whatsapp_text  # async
except Exception:
    import httpx

    WA_TOKEN = (os.getenv("WA_ACCESS_TOKEN") or "").strip()
    WA_PHONE_ID = (os.getenv("WA_PHONE_NUMBER_ID") or "").strip()
    WA_API_VER = (os.getenv("WA_API_VERSION") or "v21.0").strip()

    async def send_whatsapp_text(to_number: str, body: str) -> Dict[str, Any]:
        """Envío básico de texto por WhatsApp usando la Cloud API."""
        if not (WA_TOKEN and WA_PHONE_ID):
            log.warning("MCP/WhatsApp: tokens faltantes, no se envía mensaje")
            return {"ok": False, "error": "WA tokens faltantes"}

        url = f"https://graph.facebook.com/{WA_API_VER}/{WA_PHONE_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WA_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": body[:4096]},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json=payload)
            ok = 200 <= resp.status_code < 300
            data = resp.json() if resp.content else {}
            if not ok:
                log.warning("MCP/WhatsApp error %s: %s", resp.status_code, data)
            return {"ok": ok, "status": resp.status_code, "data": data}


# ---------------------------------------------------------------------
# Router MCP  (el prefix /mcp lo pondremos en main.py)
# ---------------------------------------------------------------------
router = APIRouter(tags=["mcp"])


@router.get("/")
async def mcp_root():
    """Información básica del servidor MCP."""
    return {
        "ok": True,
        "service": "MCP CCP",
        "time": dt.datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "host": platform.node(),
        "endpoints": ["/mcp/health", "/mcp/ping", "/mcp/ask"],
    }


@router.get("/health")
async def mcp_health():
    """Salud del servicio MCP."""
    return {
        "ok": True,
        "service": "MCP CCP",
        "time": dt.datetime.utcnow().isoformat() + "Z",
    }


@router.post("/ping")
async def mcp_ping(payload: Dict[str, Any] | None = None):
    """Ping sencillo para pruebas."""
    return {"ok": True, "echo": payload or {}}


@router.post("/ask")
async def mcp_ask(request: Request):
    """
    Endpoint para consultar el RAG.

    Body esperado:
    {
      "question": "texto del usuario",
      "wa_number": "573XXXXXXXX"  (opcional, si quieres que además se envíe por WhatsApp)
    }
    """
    data = await request.json()
    question = (data.get("question") or "").strip()
    wa_number = (data.get("wa_number") or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="question requerido")

    log.info("MCP/ask: pregunta='%s' wa_number='%s'", question, wa_number or "-")

    try:
        # Llama a tu pipeline RAG
        answer = await answer_with_rag(question)
    except Exception as e:
        log.exception("Error ejecutando RAG desde MCP")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"RAG_ERROR: {str(e)}"},
        )

    result: Dict[str, Any] = {"ok": True, "answer": answer}

    
    if wa_number:
        try:
            wa_res = await send_whatsapp_text(wa_number, answer)
            result["whatsapp"] = wa_res
        except Exception as e:
            log.exception("Error enviando WhatsApp desde MCP")
            result["whatsapp"] = {"ok": False, "error": str(e)}

    return result
