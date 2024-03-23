from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
import uvicorn
import pickle
import pandas as pd
import numpy as np

app = FastAPI()
workload_ids = [1, 7, 21, 31, 37, 50, 55, 56, 69, 71]
timeseries_models = {}
df = pd.read_csv("all_data.csv")
individual_dfs = {}
executor = ThreadPoolExecutor(max_workers=10)

for i in workload_ids:
    with open(f'../../data/timeseries_model_files/xgb_model_{i}.pkl', 'rb') as f:
        timeseries_models[i] = pickle.load(f)

for i in workload_ids:
    individual_dfs[i] = pd.read_csv(f'../../data/training_data/{i}.txt', delimiter=',')
    individual_dfs[i].drop(columns=["Unnamed: 0"], inplace=True)
    X_features = [feature for feature in individual_dfs[i] if feature != 'invocations']
    individual_dfs[i] = individual_dfs[i][X_features]


def predict_with_model(workload_id, second):
    return timeseries_models[workload_id].predict_proba(individual_dfs[workload_id][:second-1])


class DataPayload(BaseModel):
    second: int


@app.post("/predict_future_invications")
async def predict_future_invications(data: DataPayload):
    second = data.second
    futures = [executor.submit(predict_with_model, workload_id, second) for workload_id in workload_ids]
    results = [future.result() for future in futures]
    results = [results[i][-1] for i in range(len(workload_ids))]
    probabilities_of_1 = [array[1] for array in results]
    #index_of_highest_prob_of_1 = np.argmax(probabilities_of_1)
    indices_of_highest_values = np.argsort(probabilities_of_1)[-4:] # taking 4 highest values

    return {"workload_list_indices": indices_of_highest_values.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8556)