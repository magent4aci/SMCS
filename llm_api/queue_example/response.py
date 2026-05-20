import sqlite3
import time

DB_FILE = "your path to queue_example/shared/messages.db"


def process_request(request_id, request_content):
    """Process request and generate response"""
    response_content = f"Response to '{request_content}' from Server B"
    return response_content


def listen_for_requests():
    """Listen for requests and send responses"""
    while True:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, content FROM requests LIMIT 1')
            request = cursor.fetchone()

            if request:
                request_id, request_content = request
                print(f"Server B received request: {request_id} - {request_content}")

                # Process request and generate response
                response_content = process_request(request_id, request_content)

                # Insert response into database
                cursor.execute('INSERT INTO responses (id, content) VALUES (?, ?)', (request_id, response_content))
                conn.commit()

                # Remove processed request
                cursor.execute('DELETE FROM requests WHERE id = ?', (request_id,))
                conn.commit()

                print(f"Server B sent response: {response_content}")
        time.sleep(1)


def main():
    listen_for_requests()


if __name__ == "__main__":
    main()