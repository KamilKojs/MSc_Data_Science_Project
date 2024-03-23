from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
import torch
import requests
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List
import copy

app = FastAPI()
evaluation_results = {}
dl_models = {}
workload_ids = [1, 7, 21, 31, 37, 50, 55, 56, 69, 71]
image_path = "sample_image.jpg"
current_models_on_gpu = [1, 7, 21, 31]
loading_manager_url = "http://localhost:8556/predict_future_invications"

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

for workload_id in current_models_on_gpu:
    dl_models[workload_id].to("cuda:0")


class DataPayload(BaseModel):
    cols_with_1: list
    second: int


class EvaluationEndPayload(BaseModel):
    evaluation_time: float


def make_prediction(model_id: int, not_needed_models_on_gpu: List[int]):
    global current_models_on_gpu
    input_tensor = preprocess(image)
    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to('cuda:0')

    if model_id in current_models_on_gpu:
        unload_from_gpu_time = 0
        model_to_gpu_time = 0
        # Perform inference
        start_inference = time.time()
        with torch.no_grad():
            output = dl_models[model_id](input_batch)
        end_inference = time.time()
        inference_time = end_inference - start_inference

    else: #model_id not in current_models_on_gpu:
        # unload not needed model from GPU
        start_unload_from_gpu = time.time()
        not_needed_model_on_gpu = not_needed_models_on_gpu[0]
        dl_models[not_needed_model_on_gpu].to("cpu")
        end_unload_from_gpu = time.time()
        unload_from_gpu_time = end_unload_from_gpu - start_unload_from_gpu
        not_needed_models_on_gpu.pop(0)

        # move desired model to GPU
        start_model_to_gpu = time.time()
        dl_models[model_id].to("cuda:0")
        end_model_to_gpu = time.time()
        model_to_gpu_time = end_model_to_gpu - start_model_to_gpu
        current_models_on_gpu.remove(not_needed_model_on_gpu)
        current_models_on_gpu.append(model_id)

        # Perform inference
        start_inference = time.time()
        with torch.no_grad():
            output = dl_models[model_id](input_batch)
        end_inference = time.time()
        inference_time = end_inference - start_inference

    return model_to_gpu_time, inference_time, unload_from_gpu_time


def preload_models(workload_ids_to_preload: List[int]):
    global current_models_on_gpu
    # unload previous models
    for workload_id in current_models_on_gpu:
        dl_models[workload_id].to("cpu")
    # move desired models to GPU
    for workload_id in workload_ids_to_preload:
        dl_models[workload_id].to("cuda:0")

    current_models_on_gpu = workload_ids_to_preload


@app.post("/process_data")
def process_data(data: DataPayload):
    global current_models_on_gpu
    cols_with_1 = data.cols_with_1
    second = data.second

    # decide which model to preload to GPU based on "loading manager"
    payload = {
        "second": second
    }
    start_manager_time_series_time = time.time()
    response = requests.post(loading_manager_url, json=payload)
    end_manager_time_series_time = time.time()
    manager_time_series_time = end_manager_time_series_time - start_manager_time_series_time

    models_on_gpu_before_preloading = copy.deepcopy(current_models_on_gpu)

    workload_list_indices = response.json()["workload_list_indices"]
    workload_ids_to_preload = []
    for indice in workload_list_indices:     
        workload_ids_to_preload.append(workload_ids[indice])

    preload_models(workload_ids_to_preload)

    evaluation_results[second] = {}
    requested_workloads = [int(col) for col in cols_with_1]
    requested_workloads_set = set(requested_workloads)
    current_models_on_gpu_set = set(current_models_on_gpu)
    not_needed_models_on_gpu = current_models_on_gpu_set - requested_workloads_set
    not_needed_models_on_gpu = list(not_needed_models_on_gpu)
    for col in cols_with_1:
        start_time = time.time()
        # load the model and make prediction
        model_to_gpu_time, inference_time, unload_from_gpu_time = make_prediction(int(col), not_needed_models_on_gpu)
        end_time = time.time()

        # Record the time taken for the operation
        evaluation_results[second][col] = {}
        evaluation_results[second][col]["processing_time"] = end_time - start_time
        evaluation_results[second][col]["workload_id_to_preload_1"] = workload_ids_to_preload[0]
        evaluation_results[second][col]["workload_id_to_preload_2"] = workload_ids_to_preload[1]
        evaluation_results[second][col]["workload_id_to_preload_3"] = workload_ids_to_preload[2]
        evaluation_results[second][col]["workload_id_to_preload_4"] = workload_ids_to_preload[3]
        evaluation_results[second][col]["model_on_gpu_before_preloading_1"] = models_on_gpu_before_preloading[0]
        evaluation_results[second][col]["model_on_gpu_before_preloading_2"] = models_on_gpu_before_preloading[1]
        evaluation_results[second][col]["model_on_gpu_before_preloading_3"] = models_on_gpu_before_preloading[2]
        evaluation_results[second][col]["model_on_gpu_before_preloading_4"] = models_on_gpu_before_preloading[3]
        evaluation_results[second][col]["manager_time_series_time"] = manager_time_series_time
        evaluation_results[second][col]["model_to_gpu_time"] = model_to_gpu_time
        evaluation_results[second][col]["inference_time"] = inference_time
        evaluation_results[second][col]["unload_from_gpu_time"] = unload_from_gpu_time
    
    return {"message": "Data processed successfully"}


@app.post("/save_evaluation_results")
def save_results(data: EvaluationEndPayload):
    evaluation_time = data.evaluation_time
    # Save evaluation results to file
    with open("evaluation_results.txt", "a") as f:
        for second, col_times in evaluation_results.items():
            for col, metadata in col_times.items():
                f.write(f"{second},{col},{metadata['processing_time']},{metadata['workload_id_to_preload_1']},{metadata['workload_id_to_preload_2']},{metadata['workload_id_to_preload_3']},{metadata['workload_id_to_preload_4']},{metadata['model_on_gpu_before_preloading_1']},{metadata['model_on_gpu_before_preloading_2']},{metadata['model_on_gpu_before_preloading_3']},{metadata['model_on_gpu_before_preloading_4']},{metadata['manager_time_series_time']},{metadata['model_to_gpu_time']},{metadata['inference_time']},{metadata['unload_from_gpu_time']}\n")

    with open("evaluation_time.txt", "a") as f:
        f.write(f"Evaluation time (full experiment time): {evaluation_time}\n")

    return {"message": "Evaluation results saved successfuly"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8555)