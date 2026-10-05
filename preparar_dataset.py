
import os
import random
import shutil
from pathlib import Path

# ============================================================
# 1. LOCALIZAÇÃO DO TRASHNET
# ============================================================

DATASET_ORIGINAL = Path(
    "trashnet-master/data/dataset-resized"
)

# ============================================================
# 2. NOME DO NOVO DATASET
# ============================================================

DATASET_YOLO = Path("SIMR_DATASET")

# ============================================================
# 3. CLASSES DO TRASHNET
# ============================================================

CLASSES = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5
}

# ============================================================
# 4. DIVISÃO DO DATASET
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

random.seed(42)

# ============================================================
# 5. VERIFICAR SE O TRASHNET EXISTE
# ============================================================

if not DATASET_ORIGINAL.exists():

    print()
    print("ERRO!")
    print("A pasta do TrashNet não foi encontrada.")
    print()
    print("O programa procurou em:")
    print(DATASET_ORIGINAL.resolve())
    print()

    exit()

print()
print("=" * 60)
print("TRASHNET ENCONTRADO!")
print("=" * 60)

print()
print("Local:")
print(DATASET_ORIGINAL.resolve())

# ============================================================
# 6. CRIAR AS PASTAS DO YOLO
# ============================================================

for split in ["train", "val", "test"]:

    pasta_images = (
        DATASET_YOLO /
        split /
        "images"
    )

    pasta_labels = (
        DATASET_YOLO /
        split /
        "labels"
    )

    pasta_images.mkdir(
        parents=True,
        exist_ok=True
    )

    pasta_labels.mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# 7. PROCESSAR CADA CLASSE
# ============================================================

total_imagens = 0

for classe, classe_id in CLASSES.items():

    pasta_classe = (
        DATASET_ORIGINAL /
        classe
    )

    if not pasta_classe.exists():

        print()
        print(
            f"AVISO: classe '{classe}' não encontrada."
        )

        continue

    # Buscar imagens
    imagens = []

    for extensao in [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]:

        imagens.extend(
            pasta_classe.glob(extensao)
        )

    # Embaralhar
    random.shuffle(imagens)

    total = len(imagens)

    # ========================================================
    # DIVISÃO
    # ========================================================

    quantidade_train = int(
        total * TRAIN_RATIO
    )

    quantidade_val = int(
        total * VAL_RATIO
    )

    train_images = imagens[
        :quantidade_train
    ]

    val_images = imagens[
        quantidade_train:
        quantidade_train + quantidade_val
    ]

    test_images = imagens[
        quantidade_train + quantidade_val:
    ]

    conjuntos = {

        "train": train_images,

        "val": val_images,

        "test": test_images
    }

    # ========================================================
    # MOSTRAR INFORMAÇÕES
    # ========================================================

    print()
    print("-" * 60)

    print(
        f"CLASSE: {classe}"
    )

    print(
        f"Total: {total}"
    )

    print(
        f"Treinamento: {len(train_images)}"
    )

    print(
        f"Validação: {len(val_images)}"
    )

    print(
        f"Teste: {len(test_images)}"
    )

    # ========================================================
    # COPIAR IMAGENS
    # ========================================================

    for split, lista_imagens in conjuntos.items():

        for imagem in lista_imagens:

            # Criar novo nome
            novo_nome = (
                f"{classe}_{imagem.name}"
            )

            # ------------------------------------------------
            # DESTINO DA IMAGEM
            # ------------------------------------------------

            destino_imagem = (
                DATASET_YOLO /
                split /
                "images" /
                novo_nome
            )

            shutil.copy2(
                imagem,
                destino_imagem
            )

            # ------------------------------------------------
            # CRIAR LABEL
            # ------------------------------------------------

            nome_label = (
                Path(novo_nome).stem
                + ".txt"
            )

            caminho_label = (
                DATASET_YOLO /
                split /
                "labels" /
                nome_label
            )

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------
            #
            # TrashNet é classificação.
            #
            # Portanto inicialmente vamos considerar
            # que o objeto ocupa toda a imagem.
            #
            # YOLO:
            #
            # classe
            # x_centro
            # y_centro
            # largura
            # altura
            #
            # Tudo normalizado entre 0 e 1.
            # ------------------------------------------------

            with open(
                caminho_label,
                "w"
            ) as arquivo:

                arquivo.write(
                    f"{classe_id} "
                    f"0.5 0.5 1.0 1.0\n"
                )

            total_imagens += 1

# ============================================================
# 8. CRIAR DATA.YAML
# ============================================================

yaml_content = f"""path: {DATASET_YOLO.resolve()}

train: train/images
val: val/images
test: test/images

nc: 6

names:
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: trash
"""

with open(
    DATASET_YOLO / "data.yaml",
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(yaml_content)

# ============================================================
# 9. RESULTADO FINAL
# ============================================================

print()
print("=" * 60)
print("DATASET DO SIMR CRIADO COM SUCESSO!")
print("=" * 60)

print()
print(
    f"Total de imagens: {total_imagens}"
)

print()
print("Local do dataset:")

print(
    DATASET_YOLO.resolve()
)

print()
print("Estrutura criada:")

print("""
SIMR_DATASET/
│
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
""")

print("CLASSES:")

for nome, numero in CLASSES.items():

    print(
        f"{numero} = {nome}"
    )

print()
print("=" * 60)
print("PREPARAÇÃO CONCLUÍDA!")
print("=" * 60)
from ultralytics import YOLO

model = YOLO(
    "runs/detect/SIMR_YOLOv8_TrashNet/weights/last.pt"
)

print("Continuando treinamento...")

model.train(resume=True)