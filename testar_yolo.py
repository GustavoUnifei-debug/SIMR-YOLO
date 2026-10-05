from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO(
    "runs/detect/SIMR_YOLOv8_TrashNet/weights/best.pt"
)

# Testa as imagens do conjunto de teste
results = model.predict(
    source="SIMR_DATASET/test/images",
    conf=0.25,
    save=True
)

print("\n===================================")
print("TESTE DO YOLOv8 FINALIZADO")
print("===================================")
print("As imagens com as deteccoes foram salvas.")