import json

with open('./input_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

import random

# Randomly select 300 indices for test set
total_lines = len(data)
test_indices = random.sample(range(total_lines), 300)
test_indices.sort()  # Sort indices for consistency

# Split data into test and train sets
test_data = [data[i] for i in test_indices]
train_data = [data[i] for i in range(total_lines) if i not in test_indices]

# Save test indices
with open('test_ids.json', 'w') as f:
    json.dump(test_indices, f)
    
# Save test and train data
with open('test.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')
        
with open('train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')
