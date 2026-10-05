import cv2
from pathlib import Path


# =========================================================
# CONFIGURACAO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLASSE = "glass"

PASTA_IMAGENS = (
    BASE_DIR
    / "dataset_simr_real"
    / "imagens"
    / CLASSE
)

PASTA_LABELS = (
    BASE_DIR
    / "dataset_simr_real"
    / "labels"
    / CLASSE
)


# =========================================================
# IMAGENS
# =========================================================

imagens = sorted(
    PASTA_IMAGENS.glob("*.jpg")
)

if not imagens:
    print("Nenhuma imagem encontrada.")
    exit()


print("=" * 55)
print("SIMR - VERIFICACAO DAS ANOTACOES")
print("=" * 55)

print(f"Classe: {CLASSE.upper()}")
print(f"Imagens: {len(imagens)}")
print()
print("N = proxima imagem")
print("Q = sair")


# =========================================================
# VERIFICACAO
# =========================================================

indice = 0

while indice < len(imagens):

    caminho_imagem = imagens[indice]

    imagem = cv2.imread(
        str(caminho_imagem)
    )

    if imagem is None:
        indice += 1
        continue

    altura, largura = imagem.shape[:2]

    caminho_label = (
        PASTA_LABELS
        / f"{caminho_imagem.stem}.txt"
    )

    # =====================================================
    # LER LABEL YOLO
    # =====================================================

    if caminho_label.exists():

        with open(
            caminho_label,
            "r",
            encoding="utf-8"
        ) as arquivo:

            linhas = arquivo.readlines()

        for linha in linhas:

            dados = linha.strip().split()

            if len(dados) != 5:
                continue

            classe_id = int(dados[0])

            centro_x = float(dados[1])
            centro_y = float(dados[2])

            largura_box = float(dados[3])
            altura_box = float(dados[4])

            # =============================================
            # YOLO -> PIXELS
            # =============================================

            cx = centro_x * largura
            cy = centro_y * altura

            w = largura_box * largura
            h = altura_box * altura

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)

            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # =============================================
            # DESENHAR
            # =============================================

            cv2.rectangle(
                imagem,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                imagem,
                "GLASS",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    else:

        cv2.putText(
            imagem,
            "SEM ANOTACAO",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =====================================================
    # INFORMACOES
    # =====================================================

    cv2.putText(
        imagem,
        f"{indice + 1}/{len(imagens)}",
        (20, altura - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SIMR - Verificacao YOLO",
        imagem
    )

    tecla = cv2.waitKey(0) & 0xFF

    if tecla == ord("n"):

        indice += 1

    elif tecla == ord("q"):

        break


cv2.destroyAllWindows()

print("Verificacao finalizada.")