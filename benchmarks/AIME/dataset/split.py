import json

train_data = []

with open('./AIME_1983_2024.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if int(data['url'][:4]) < 2024:
            train_data.append(data)

with open('./AIME_1983_2023.jsonl', 'w') as f:
    for data in train_data:
        f.write(json.dumps(data) + '\n')
