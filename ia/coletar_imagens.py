import cv2
from pathlib import Path
from datetime import datetime


# =========================================================
# CONFIGURACAO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "dataset_simr_real"
    / "imagens"
)

CLASSES = {
    ord("1"): "glass",
    ord("2"): "metal",
    ord("3"): "plastic",
    ord("4"): "paper",
    ord("5"): "cardboard"
}


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERRO: Nao foi possivel abrir a camera.")
    exit()


classe_atual = None
contador = 0


print("=" * 55)
print("SIMR - COLETA DE IMAGENS")
print("=" * 55)

print("1 = GLASS")
print("2 = METAL")
print("3 = PLASTIC")
print("4 = PAPER")
print("5 = CARDBOARD")
print()
print("ESPACO = salvar imagem")
print("Q = sair")


# =========================================================
# LOOP
# =========================================================

while True:

    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem.")
        break

    altura, largura = frame.shape[:2]


    # =====================================================
    # ROI CENTRAL
    # =====================================================

    roi_largura = int(largura * 0.50)
    roi_altura = int(altura * 0.65)

    x1 = (largura - roi_largura) // 2
    y1 = (altura - roi_altura) // 2

    x2 = x1 + roi_largura
    y2 = y1 + roi_altura


    # =====================================================
    # DESENHAR ROI
    # =====================================================

    frame_exibicao = frame.copy()

    cv2.rectangle(
        frame_exibicao,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame_exibicao,
        "OBJETO DENTRO DESTA AREA",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


    # =====================================================
    # INFORMACOES
    # =====================================================

    if classe_atual is None:
        texto_classe = "SELECIONE 1-5"
    else:
        texto_classe = classe_atual.upper()

    cv2.rectangle(
        frame_exibicao,
        (0, 0),
        (largura, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame_exibicao,
        "SIMR - COLETA DO DATASET",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_exibicao,
        f"Classe: {texto_classe}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_exibicao,
        f"Fotos nesta sessao: {contador}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # =====================================================
    # MOSTRAR
    # =====================================================

    cv2.imshow(
        "SIMR - Coleta de Imagens",
        frame_exibicao
    )

    tecla = cv2.waitKey(1) & 0xFF


    # =====================================================
    # SELECIONAR CLASSE
    # =====================================================

    if tecla in CLASSES:

        classe_atual = CLASSES[tecla]
        contador = 0

        print()
        print(
            f"Classe selecionada: "
            f"{classe_atual.upper()}"
        )


    # =====================================================
    # ESPACO = SALVAR
    # =====================================================

    elif tecla == 32:

        if classe_atual is None:

            print(
                "Selecione uma classe primeiro."
            )

            continue

        pasta = DATASET_DIR / classe_atual

        pasta.mkdir(
            parents=True,
            exist_ok=True
        )

        # Salvar a imagem COMPLETA.
        # A ROI serve como guia para posicionar o objeto.
        imagem = frame.copy()

        horario = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        nome = (
            f"{classe_atual}_{horario}.jpg"
        )

        caminho = pasta / nome

        cv2.imwrite(
            str(caminho),
            imagem
        )

        contador += 1

        print(
            f"Imagem salva: {nome}"
        )


    # =====================================================
    # Q = SAIR
    # =====================================================

    elif tecla == ord("q"):
        break


# =========================================================
# FINALIZAR
# =========================================================

camera.release()
cv2.destroyAllWindows()

print()
print("Coleta finalizada.")