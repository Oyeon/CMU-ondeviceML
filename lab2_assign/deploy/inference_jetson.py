from ultralytics import YOLO
import torch

model_config = ['best.pt']
input_size = [640]
inference_batch_size = 1
index = 0

MODEL = model_config[index] 
INPUT_SIZE = input_size[index]

model = YOLO(MODEL)

DATA_YAML = 'data.yaml'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = 'results'
metrics = model.val(data=DATA_YAML, imgsz=INPUT_SIZE, device=device, save_json=True, plots=True, project=RESULTS_DIR)

print('done')
