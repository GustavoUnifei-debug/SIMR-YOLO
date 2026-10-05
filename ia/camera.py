import cv2
from ultralytics import YOLO
from pathlib import Path


# =========================================================
# CONFIGURACAO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELO = BASE_DIR / "modelo" / "best.pt"

CONFIANCA_MINIMA = 0.50


# =========================================================
# CARREGAR MODELO
# =========================================================

print("=" * 55)
print("SIMR - CAPTURA E ANALISE DE MATERIAL")
print("=" * 55)

print("Carregando YOLOv8...")

model = YOLO(str(MODELO))

print("Modelo carregado.")
print()
print("ESPACO = capturar e analisar")
print("Q = encerrar")


# =========================================================
# ABRIR CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERRO: Nao foi possivel abrir a camera.")
    exit()


# Resultado inicial
material = "AGUARDANDO CAPTURA"
confianca = 0.0


# =========================================================
# LOOP DA CAMERA
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
    # ESCURECER AREA EXTERNA
    # =====================================================

    frame_exibicao = frame.copy()

    mascara = frame.copy()
    mascara[:] = (0, 0, 0)

    frame_escuro = cv2.addWeighted(
        frame,
        0.35,
        mascara,
        0.65,
        0
    )

    frame_escuro[y1:y2, x1:x2] = (
        frame[y1:y2, x1:x2]
    )

    frame_exibicao = frame_escuro


    # =====================================================
    # AREA DE ANALISE
    # =====================================================

    cv2.rectangle(
        frame_exibicao,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame_exibicao,
        "AREA DE ANALISE",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # =====================================================
    # PAINEL
    # =====================================================

    cv2.rectangle(
        frame_exibicao,
        (0, 0),
        (largura, 135),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame_exibicao,
        "SIMR - CAPTURA CONTROLADA",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_exibicao,
        f"Material: {material}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_exibicao,
        f"Confianca: {confianca:.1f}%",
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_exibicao,
        "ESPACO = ANALISAR | Q = SAIR",
        (20, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # =====================================================
    # MOSTRAR CAMERA
    # =====================================================

    cv2.imshow(
        "SIMR - Captura Controlada",
        frame_exibicao
    )


    tecla = cv2.waitKey(1) & 0xFF


    # =====================================================
    # ESPACO = CAPTURAR E ANALISAR
    # =====================================================

    if tecla == 32:

        print()
        print("Capturando imagem...")

        # Recortar exatamente a ROI
        roi = frame[y1:y2, x1:x2].copy()

        print("Analisando com YOLOv8...")

        resultados = model.predict(
            source=roi,
            conf=CONFIANCA_MINIMA,
            verbose=False
        )

        resultado = resultados[0]

        material = "NAO IDENTIFICADO"
        confianca = 0.0

        if (
            resultado.boxes is not None
            and len(resultado.boxes) > 0
        ):

            melhor_confianca = -1
            melhor_classe = None

            for box in resultado.boxes:

                conf = float(box.conf[0])
                classe_id = int(box.cls[0])

                if conf > melhor_confianca:

                    melhor_confianca = conf
                    melhor_classe = classe_id

            if melhor_classe is not None:

                material = (
                    resultado.names[
                        melhor_classe
                    ].upper()
                )

                confianca = (
                    melhor_confianca * 100
                )

        print("------------------------------")
        print("RESULTADO SIMR")
        print("------------------------------")
        print(f"Material: {material}")
        print(f"Confianca: {confianca:.1f}%")
        print("------------------------------")


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

print("Camera finalizada.")