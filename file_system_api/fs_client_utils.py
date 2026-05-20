import fcntl
import time
import uuid
import json
import os


class FileSystemClient:
    def __init__(self, model_name, request_folder, response_folder, fps=0.1):
        self.model_name = model_name
        self.request_folder = request_folder
        self.response_folder = response_folder
        self.client_id = 'client' + str(uuid.uuid4()).split('-')[0]
        self.fps = fps

    def send_and_get_response(self, messages, max_tokens, temperature):
        request_id = self.client_id + '-' + self.model_name + '-' + str(uuid.uuid4())
        request_id = self.send_request(request_id, messages, max_tokens, temperature)
        return self.wait_for_response(request_id)

    def send_request(self, request_id, messages, max_tokens, temperature):
        data = {"messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature}
        # write the request
        with open(os.path.join(self.request_folder, str(request_id) + '.json'), 'w') as f:
            json.dump(data, f)
        return request_id

    def wait_for_response(self, request_id):
        while True:
            all_response_file = os.listdir(self.response_folder)
            if str(request_id) + '.json' in all_response_file:
                # Remove the processed response
                with open(os.path.join(self.response_folder, str(request_id) + '.json'), 'r') as f:
                    res = json.load(f)
                os.remove(os.path.join(self.response_folder, str(request_id) + '.json'))
                return res
            time.sleep(self.fps)

    # Function to acquire file lock
    def acquire_lock(self, file_path):
        lock_file = open(file_path, 'a')  # Open file in append mode
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # Lock file exclusively
        return lock_file

    # Function to release file lock
    def release_lock(self, lock_file):
        fcntl.flock(lock_file, fcntl.LOCK_UN)  # Unlock file
        lock_file.close()


if __name__ == '__main__':
    client = FileSystemClient("Qwen2.5-Math-7B-Instruct",
                              "your path to queue_example/shared/request.json",
                              "your path to queue_example/shared/response.json",
                              fps=0.1)
    prompt = 'Please introduce yourself and calculate 25*88'
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.send_and_get_response(messages, max_tokens=2048, temperature=0.7)
    print(response)