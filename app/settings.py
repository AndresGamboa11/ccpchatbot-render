# app/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # --- WhatsApp Cloud API (ES + fallback EN) ---
    TOKEN_WA: str = Field(default="", alias="WA_ACCESS_TOKEN")
    ID_NUMERO_WA: str = Field(default="", alias="WA_PHONE_NUMBER_ID")
    TOKEN_VERIFICACION_WA: str = Field(default="", alias="WA_VERIFY_TOKEN")
    VERSION_WA: str = Field(default="v21.0", alias="WA_API_VERSION")

    # --- Groq (Gemma) ---
    CLAVE_GROQ: str = Field(default="", alias="GROQ_API_KEY")
    MODELO_GROQ: str = Field(default="gemma2-9b-it", alias="GROQ_MODEL")

    # --- Qdrant Cloud ---
    URL_QDRANT: str = Field(default="", alias="QDRANT_URL")
    CLAVE_API_QDRANT: str = Field(default="", alias="QDRANT_API_KEY")
    COLECCION_QDRANT: str = Field(default="ccp_docs", alias="QDRANT_COLLECTION")

    # --- Afinado general ---
    TAM_EMBED: int = 384
    TIMEOUT_HTTP: int = 45

    class Config:
        extra = "ignore"
        env_file = ".env"
        env_file_encoding = "utf-8"

def get_settings() -> Settings:
    return Settings()  # lee .env automáticamente
