import pandas as pd
import requests
import time

predict_url = "http://localhost:8556/predict_future_invications"

second = 950403
        
payload = {
    "second": second
}
response = requests.post(predict_url, json=payload)
print(response.json()["workload_list_index"])