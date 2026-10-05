import cv2
from pathlib import Path


# =========================================================
# CONFIGURACAO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLASSE = "metal"

CLASSES = {
    "glass": 0,
    "metal": 1,
    "plastic": 2,
    "paper": 3,
    "cardboard": 4
}

CLASSE_ID = CLASSES[CLASSE]

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

PASTA_LABELS.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# BUSCAR SOMENTE IMAGENS SEM ANOTACAO
# =========================================================

todas_imagens = []

for extensao in ["*.jpg", "*.jpeg", "*.png"]:
    todas_imagens.extend(
        PASTA_IMAGENS.glob(extensao)
    )

todas_imagens = sorted(todas_imagens)

imagens = []

for imagem in todas_imagens:

    label = (
        PASTA_LABELS
        / f"{imagem.stem}.txt"
    )

    if not label.exists():
        imagens.append(imagem)


# =========================================================
# INFORMACOES
# =========================================================

print("=" * 60)
print("SIMR - ANOTADOR YOLO")
print("=" * 60)

print(f"Classe: {CLASSE.upper()}")
print(f"Total de imagens: {len(todas_imagens)}")
print(
    f"Ja anotadas: "
    f"{len(todas_imagens) - len(imagens)}"
)
print(f"Faltando anotar: {len(imagens)}")
print()


if len(imagens) == 0:

    print("Todas as imagens ja possuem anotacao.")
    exit()


print("CONTROLES")
print("Mouse = desenhar caixa")
print("S = salvar e ir para proxima")
print("R = refazer caixa")
print("N = pular imagem")
print("Q = sair")


# =========================================================
# VARIAVEIS
# =========================================================

desenhando = False
caixa_pronta = False

x_inicio = 0
y_inicio = 0
x_fim = 0
y_fim = 0


# =========================================================
# MOUSE
# =========================================================

def desenhar_caixa(event, x, y, flags, param):

    global desenhando
    global caixa_pronta

    global x_inicio
    global y_inicio
    global x_fim
    global y_fim

    if event == cv2.EVENT_LBUTTONDOWN:

        desenhando = True
        caixa_pronta = False

        x_inicio = x
        y_inicio = y
        x_fim = x
        y_fim = y

    elif event == cv2.EVENT_MOUSEMOVE:

        if desenhando:
            x_fim = x
            y_fim = y

    elif event == cv2.EVENT_LBUTTONUP:

        desenhando = False

        x_fim = x
        y_fim = y

        caixa_pronta = True


# =========================================================
# JANELA
# =========================================================

cv2.namedWindow(
    "SIMR - Anotacao YOLO"
)

cv2.setMouseCallback(
    "SIMR - Anotacao YOLO",
    desenhar_caixa
)


# =========================================================
# PROCESSAR IMAGENS
# =========================================================

indice = 0


while indice < len(imagens):

    caminho_imagem = imagens[indice]

    imagem_original = cv2.imread(
        str(caminho_imagem)
    )

    if imagem_original is None:

        print(
            f"Erro ao abrir: "
            f"{caminho_imagem.name}"
        )

        indice += 1
        continue


    altura, largura = (
        imagem_original.shape[:2]
    )

    caixa_pronta = False
    desenhando = False

    x_inicio = 0
    y_inicio = 0
    x_fim = 0
    y_fim = 0


    while True:

        imagem = imagem_original.copy()


        # =================================================
        # DESENHAR CAIXA
        # =================================================

        if desenhando or caixa_pronta:

            cv2.rectangle(
                imagem,
                (x_inicio, y_inicio),
                (x_fim, y_fim),
                (0, 255, 0),
                2
            )


        # =================================================
        # PAINEL
        # =================================================

        cv2.rectangle(
            imagem,
            (0, 0),
            (largura, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            imagem,
            (
                f"{CLASSE.upper()} | "
                f"Restante "
                f"{indice + 1}/{len(imagens)}"
            ),
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            imagem,
            "Mouse: caixa | S: salvar | R: refazer | Q: sair",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


        cv2.imshow(
            "SIMR - Anotacao YOLO",
            imagem
        )


        tecla = cv2.waitKey(20) & 0xFF


        # =================================================
        # SALVAR
        # =================================================

        if tecla == ord("s"):

            if not caixa_pronta:

                print(
                    "Desenhe uma caixa antes de salvar."
                )

                continue


            xmin = min(
                x_inicio,
                x_fim
            )

            xmax = max(
                x_inicio,
                x_fim
            )

            ymin = min(
                y_inicio,
                y_fim
            )

            ymax = max(
                y_inicio,
                y_fim
            )


            largura_caixa = (
                xmax - xmin
            )

            altura_caixa = (
                ymax - ymin
            )


            # Impedir caixa vazia
            if (
                largura_caixa <= 2
                or altura_caixa <= 2
            ):

                print(
                    "Caixa muito pequena. Refaca."
                )

                continue


            # =================================================
            # CONVERTER PARA YOLO
            # =================================================

            centro_x = (
                xmin
                + largura_caixa / 2
            ) / largura

            centro_y = (
                ymin
                + altura_caixa / 2
            ) / altura

            largura_yolo = (
                largura_caixa
                / largura
            )

            altura_yolo = (
                altura_caixa
                / altura
            )


            # =================================================
            # SALVAR TXT
            # =================================================

            caminho_label = (
                PASTA_LABELS
                / f"{caminho_imagem.stem}.txt"
            )

            with open(
                caminho_label,
                "w",
                encoding="utf-8"
            ) as arquivo:

                arquivo.write(
                    f"{CLASSE_ID} "
                    f"{centro_x:.6f} "
                    f"{centro_y:.6f} "
                    f"{largura_yolo:.6f} "
                    f"{altura_yolo:.6f}\n"
                )


            print(
                f"Salvo: "
                f"{caminho_label.name}"
            )

            indice += 1
            break


        # =================================================
        # REFAZER
        # =================================================

        elif tecla == ord("r"):

            caixa_pronta = False
            desenhando = False

            x_inicio = 0
            y_inicio = 0
            x_fim = 0
            y_fim = 0


        # =================================================
        # PULAR
        # =================================================

        elif tecla == ord("n"):

            print(
                f"Pulada: "
                f"{caminho_imagem.name}"
            )

            indice += 1
            break


        # =================================================
        # SAIR
        # =================================================

        elif tecla == ord("q"):

            cv2.destroyAllWindows()

            print()
            print("Anotacao interrompida.")

            exit()


# =========================================================
# FINALIZAR
# =========================================================

cv2.destroyAllWindows()

print()
print("=" * 60)
print("IMAGENS PENDENTES PROCESSADAS")
print("=" * 60)