# app/mcp_server.py
"""
MCP "ligero" expuesto por HTTP:
- POST /mcp/tool  { "name": "consulta_ccp", "input": "texto" }
  → ejecuta herramientas del bot (por ahora: consulta_ccp = RAG)
"""
# app/mcp_server.py
from typing import Any, Dict, Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import os, platform, datetime as dt

# ---------------- Dependencias de tu proyecto ----------------
from app.rag import answer_with_rag

# settings (opcional)
try:
    from app.settings import get_settings
    S = get_settings()
except Exception:
    S = None

# Si tienes un cliente WhatsApp propio, úsalo; si no, fallback simple
try:
    from app.whatsapp import send_whatsapp_text  # async
except Exception:
    import httpx
    WA_TOKEN = os.getenv("WA_ACCESS_TOKEN", "")
    WA_PHONE_ID = os.getenv("WA_PHONE_NUMBER_ID", "")
    WA_API_VER = os.getenv("WA_API_VERSION", "v21.0")

    async def send_whatsapp_text(to_number: str, body: str) -> Dict[str, Any]:
        if not (WA_TOKEN and WA_PHONE_ID):
            return {"ok": False, "error": "WA tokens faltantes"}
        url = f"https://graph.facebook.com/{WA_API_VER}/{WA_PHONE_ID}/messages"
        headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": body[:4096]},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            ok = 200 <= r.status_code < 300
            return {"ok": ok, "status": r.status_code, "data": r.json() if r.content else {}}

# ---------------- Router MCP ----------------
router = APIRouter(prefix="/mcp", tags=["mcp"])

@router.get("/health")
async def mcp_health():
    """Salud del servicio MCP."""
    return {
        "ok": True,
        "service": "MCP CCP",
        "time": dt.datetime.utcnow().isoformat() + "Z",
        "python": platform.python_version(),
        "host": platform.node(),
    }

@router.post("/ping")
async def mcp_ping(payload: Optional[Dict[str, Any]] = None):
    """Ping sencillo para pruebas."""
    return {"ok": True, "echo": payload or {}}

@router.post("/ask")
async def mcp_ask(request: Request):
    """
    Endpoint MCP para consultar el RAG.
    Body esperado:
    {
      "question": "texto del usuario",
      "wa_number": "573XXXXXXXX (opcional, si quieres que además se envíe por WhatsApp)"
    }
    """
    data = await request.json()
    question = (data.get("question") or "").strip()
    wa_number = (data.get("wa_number") or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="question requerido")

    try:
        # Llama a tu pipeline RAG
        answer = await answer_with_rag(question)
    except Exception as e:
        # No rompas el server si RAG falla
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"RAG_ERROR: {str(e)}"},
        )

    result = {"ok": True, "answer": answer}

    # Enviar por WhatsApp (opcional)
    if wa_number:
        try:
            wa_res = await send_whatsapp_text(wa_number, answer)
            result["whatsapp"] = wa_res
        except Exception as e:
            result["whatsapp"] = {"ok": False, "error": str(e)}

    return result
