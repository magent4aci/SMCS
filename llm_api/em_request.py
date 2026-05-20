import requests
import json
from tqdm import tqdm


def get_completion():
    headers = {'Content-Type': 'application/json'}
    passages = [
        "I love you very much",
        "I like you.",
        "I hate you very much",
        "I LOVE you.",
        "I hate you.",
    ]
    tasks = ['Sentences with similar sentiment to this sentence']
    batch_size = 2
    data = {"sentences": passages,
            'tasks': tasks*5,
            'batch_size': batch_size,
            'max_length': 8192}
    # response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
    response = requests.post(url='http://172.30.40.9:6006', headers=headers, data=json.dumps(data))
    return response.json()


if __name__ == '__main__':
    for i in tqdm(range(2000)):
        print(get_completion())