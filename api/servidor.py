
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import tempfile
import os

from ia.detectar import detectar_material


# =========================================================
# CONFIGURACAO DO PROJETO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INTERFACE = BASE_DIR / "interface" / "index.html"

# Tamanho maximo permitido: 10 MB
TAMANHO_MAXIMO = 10 * 1024 * 1024

# Tipos de imagem permitidos
TIPOS_PERMITIDOS = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="SIMR API",
    description=(
        "API do Sistema de Identificacao "
        "de Materiais Reciclaveis"
    ),
    version="1.1"
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
async def detectar(
    arquivo: UploadFile = File(...)
):

    caminho = None

    try:

        # -------------------------------------------------
        # Verificar tipo do arquivo
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
        # Ler imagem
        # -------------------------------------------------

        conteudo = await arquivo.read()

        # -------------------------------------------------
        # Verificar tamanho
        # -------------------------------------------------

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
        # Criar arquivo temporario
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

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extensao
        ) as temp:

            temp.write(conteudo)
            caminho = temp.name

        # -------------------------------------------------
        # Executar YOLOv8 V2
        # -------------------------------------------------

        resultado = detectar_material(caminho)

        # -------------------------------------------------
        # Material detectado
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
                "altura": resultado["altura"]
            }

        # -------------------------------------------------
        # Nenhuma deteccao
        # -------------------------------------------------

        return {
            "sucesso": False,
            "material": None,
            "confianca": 0,
            "bbox": None,
            "largura": resultado["largura"],
            "altura": resultado["altura"]
        }

    except Exception as erro:

        print(
            "Erro durante a deteccao:",
            erro
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
        # Apagar arquivo temporario
        # -------------------------------------------------

        if (
            caminho is not None
            and os.path.exists(caminho)
        ):

            os.remove(caminho)
