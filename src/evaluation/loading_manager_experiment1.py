from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
import uvicorn
import pickle
import pandas as pd

app = FastAPI()
workload_ids = [1, 7, 21, 31, 37, 50, 55, 56, 69, 71]
timeseries_models = {}
df = pd.read_csv("all_data.csv")
individual_dfs = {}
executor = ThreadPoolExecutor(max_workers=10)

for i in workload_ids:
    with open(f'../../data/timseries_model_files/xgb_model_{i}.pkl', 'rb') as f:
        timeseries_models[i] = pickle.load(f)

for i in workload_ids:
    individual_dfs[i] = df[["second", f'{i}']]


def predict_with_model(model, data, second):
    return model.predict(data[:second])


class DataPayload(BaseModel):
    second: int


@app.post("/predict_future_invications")
async def predict_future_invications(data: DataPayload):
    second = data.second
    futures = [executor.submit(predict_with_model, model, data, second) for model in models]
    results = [future.result() for future in futures]

    return {"predictions": results}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8556)