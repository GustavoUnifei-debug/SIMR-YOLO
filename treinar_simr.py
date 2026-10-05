from ultralytics import YOLO

print("=" * 60)
print("SIMR - TREINAMENTO YOLOv8")
print("=" * 60)

# Carrega o modelo YOLOv8 Nano pré-treinado
model = YOLO("yolov8n.pt")

print("\nModelo carregado.")
print("Iniciando treinamento...\n")

results = model.train(
    data="SIMR_DATASET/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="SIMR_YOLOv8_TrashNet",
    device="cpu",
    save=True,
    verbose=True
)

print("\n" + "=" * 60)
print("TREINAMENTO FINALIZADO")
print("=" * 60)
from ultralytics import YOLO

print("=" * 60)
print("SIMR - CONTINUANDO TREINAMENTO YOLOv8")
print("=" * 60)

model = YOLO("runs/detect/SIMR_YOLOv8_TrashNet/weights/last.pt")

print("\nContinuando de onde parou...\n")

model.train(resume=True)

print("\n" + "=" * 60)
print("TREINAMENTO FINALIZADO")
print("=" * 60)