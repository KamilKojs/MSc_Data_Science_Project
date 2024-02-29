import pandas as pd
import requests
import time


def main():
    start_time = time.time()
    predict_url = "http://localhost:8555/process_data"
    save_evaluation_url = "http://localhost:8555/save_evaluation_results"
    df = pd.read_csv("evaluation_data.csv")

    for index, row in df.iterrows():
        cols_with_1 = row[row == 1].index.tolist()
        second = int(row['second'])
        
        payload = {
            "cols_with_1": cols_with_1,
            "second": second
        }
        response = requests.post(predict_url, json=payload)
    
    print("All requests processed. Evaluation finished")
    end_time = time.time()
    payload = {
        "evaluation_time": end_time - start_time
    }
    response = requests.post(save_evaluation_url, json=payload)
    print(response.text)


if __name__ == "__main__":
    main()