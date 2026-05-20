import requests
import json
from tqdm import tqdm


def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
    ]
    data = {"messages": messages, 'temperature': 0.7, 'max_tokens': 32678, 'logprobs': 1}
    # response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
    response = requests.post(url='http://0.0.0.0:6006', headers=headers, data=json.dumps(data))
    return response.json()[0]['response']


if __name__ == '__main__':
    for i in tqdm(range(200)):
        print(get_completion('How many zeroes are at the end of $42!$ (42 factorial)?  (Reminder: The number $n!$ is the product of the integers from 1 to $n$.  For example, $5!=5\\cdot 4\\cdot3\\cdot2\\cdot 1= 120$.'))