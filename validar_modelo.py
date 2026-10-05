from ultralytics import YOLO

print("=" * 60)
print("SIMR - VALIDACAO DO MODELO YOLOv8")
print("=" * 60)

# Carrega o melhor modelo obtido no treinamento
model = YOLO(
    "runs/detect/SIMR_YOLOv8_TrashNet/weights/best.pt"
)

print("\nModelo carregado com sucesso.")
print("Iniciando avaliacao no conjunto de teste...\n")

# Avaliacao
metrics = model.val(
    data="SIMR_DATASET/data.yaml",
    split="test",
    imgsz=640,
    device="cpu",
    plots=True
)

print("\n" + "=" * 60)
print("RESULTADOS DO MODELO")
print("=" * 60)

print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
print(f"mAP50:        {metrics.box.map50:.4f}")
print(f"mAP50-95:     {metrics.box.map:.4f}")

print("=" * 60)
print("VALIDACAO FINALIZADA")
print("=" * 60)