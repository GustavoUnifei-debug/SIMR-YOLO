
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from time import perf_counter
import tempfile
import logging
import os

from ia.detectar import detectar_material


# =========================================================
# CONFIGURACAO DO PROJETO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INTERFACE = BASE_DIR / "interface" / "index.html"

TAMANHO_MAXIMO = 10 * 1024 * 1024

TIPOS_PERMITIDOS = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

# Configurar logs para terminal e Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SIMR")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="SIMR API",
    description=(
        "API do Sistema de Identificacao "
        "de Materiais Reciclaveis"
    ),
    version="1.2"
)


# =========================================================
# INTERFACE
# =========================================================

@app.get("/")
def inicio():
    return FileResponse(INTERFACE)


# =========================================================
# STATUS DO SISTEMA
# =========================================================

@app.get("/status")
def status():
    return {
        "status": "OK",
        "sistema": "SIMR",
        "modelo": "YOLOv8 V2",
        "classes": [
            "cardboard",
            "glass",
            "metal",
            "paper",
            "plastic"
        ]
    }


# =========================================================
# DETECCAO DE MATERIAIS
# =========================================================

@app.post("/detectar")
async def detectar(arquivo: UploadFile = File(...)):

    inicio_total = perf_counter()
    caminho = None

    try:

        # -------------------------------------------------
        # VALIDAR TIPO
        # -------------------------------------------------

        if arquivo.content_type not in TIPOS_PERMITIDOS:
            return JSONResponse(
                status_code=400,
                content={
                    "sucesso": False,
                    "erro": (
                        "Formato de arquivo nao permitido. "
                        "Utilize JPG, PNG ou WEBP."
                    )
                }
            )

        # -------------------------------------------------
        # LER IMAGEM
        # -------------------------------------------------

        inicio_leitura = perf_counter()

        conteudo = await arquivo.read()

        tempo_leitura = perf_counter() - inicio_leitura

        if len(conteudo) > TAMANHO_MAXIMO:
            return JSONResponse(
                status_code=413,
                content={
                    "sucesso": False,
                    "erro": (
                        "Arquivo muito grande. "
                        "O limite e 10 MB."
                    )
                }
            )

        # -------------------------------------------------
        # CRIAR ARQUIVO TEMPORARIO
        # -------------------------------------------------

        extensao = os.path.splitext(
            arquivo.filename or ""
        )[1].lower()

        if extensao not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        }:
            extensao = ".jpg"

        inicio_arquivo = perf_counter()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extensao
        ) as temp:

            temp.write(conteudo)
            caminho = temp.name

        tempo_arquivo = perf_counter() - inicio_arquivo

        # -------------------------------------------------
        # EXECUTAR YOLOv8 V2
        # -------------------------------------------------

        inicio_ia = perf_counter()

        resultado = detectar_material(caminho)

        tempo_ia = perf_counter() - inicio_ia

        tempo_total = perf_counter() - inicio_total

        # -------------------------------------------------
        # REGISTRAR DESEMPENHO
        # -------------------------------------------------

        logger.info(
            "[SIMR] Leitura=%.3fs | "
            "Arquivo=%.3fs | "
            "IA=%.3fs | "
            "Total=%.3fs | "
            "Imagem=%.1fKB",
            tempo_leitura,
            tempo_arquivo,
            tempo_ia,
            tempo_total,
            len(conteudo) / 1024
        )

        # -------------------------------------------------
        # MATERIAL DETECTADO
        # -------------------------------------------------

        if resultado["detectado"]:

            return {
                "sucesso": True,
                "material": resultado["classe"],
                "confianca": round(
                    resultado["confianca"] * 100,
                    2
                ),
                "bbox": resultado["bbox"],
                "largura": resultado["largura"],
                "altura": resultado["altura"],
                "tempo_processamento": round(
                    tempo_total,
                    3
                ),
                "tempo_ia": round(
                    tempo_ia,
                    3
                )
            }

        # -------------------------------------------------
        # NENHUMA DETECCAO
        # -------------------------------------------------

        return {
            "sucesso": False,
            "material": None,
            "confianca": 0,
            "bbox": None,
            "largura": resultado["largura"],
            "altura": resultado["altura"],
            "tempo_processamento": round(
                tempo_total,
                3
            ),
            "tempo_ia": round(
                tempo_ia,
                3
            )
        }

    except Exception:

        logger.exception(
            "[SIMR] Erro durante a deteccao"
        )

        return JSONResponse(
            status_code=500,
            content={
                "sucesso": False,
                "erro": (
                    "Erro interno durante "
                    "a analise da imagem."
                )
            }
        )

    finally:

        # -------------------------------------------------
        # APAGAR ARQUIVO TEMPORARIO
        # -------------------------------------------------

        if caminho is not None:
            try:
                os.remove(caminho)
            except OSError:
                logger.warning(
                    "[SIMR] Nao foi possivel apagar "
                    "o arquivo temporario."
                )
