from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

app = FastAPI()
evaluation_results = {}
dl_models = {}
workload_ids = [1, 7, 21, 31, 37, 50, 55, 56, 69, 71]
image_path = "sample_image.jpg"
current_model_on_gpu = 1

image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for i in workload_ids:
    dl_models[i] = models.resnet152()
    dl_models[i].load_state_dict(torch.load(f'resnet152_weights_{i}.pth'))
    dl_models[i].eval()

dl_models[current_model_on_gpu].to("cuda:0")


class DataPayload(BaseModel):
    cols_with_1: list
    second: int


class EvaluationEndPayload(BaseModel):
    evaluation_time: float


def make_prediction(model_id: int):
    global current_model_on_gpu
    input_tensor = preprocess(image)
    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    if current_model_on_gpu != model_id:
        # unload previous model
        dl_models[current_model_on_gpu].to("cpu")
        # move desired model to GPU
        dl_models[model_id].to("cuda:0")
        current_model_on_gpu = model_id

    # Perform inference
    with torch.no_grad():
        output = dl_models[model_id](input_batch)


@app.post("/process_data")
def process_data(data: DataPayload):
    cols_with_1 = data.cols_with_1
    second = data.second

    evaluation_results[second] = {}
    for col in cols_with_1:
        start_time = time.time()
        # load the model and make prediction
        make_prediction(int(col))
        end_time = time.time()

        # Record the time taken for the operation
        evaluation_results[second][col] = end_time - start_time
    
    return {"message": "Data processed successfully"}


@app.post("/save_evaluation_results")
def save_results(data: EvaluationEndPayload):
    evaluation_time = data.evaluation_time
    # Save evaluation results to file
    with open("evaluation_results.txt", "a") as f:
        for second, col_times in evaluation_results.items():
            for col, time in col_times.items():
                f.write(f"{second},{col},{time}\n")

    with open("evaluation_time.txt", "a") as f:
        f.write(f"Evaluation time (full experiment time): {evaluation_time}\n")

    return {"message": "Evaluation results saved successfuly"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8555)