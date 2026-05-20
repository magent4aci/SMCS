import sqlite3
import time
import uuid
import json

DB_FILE = "./shared/messages.db"



def init_db():
    """Initialize database"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY,
                content TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id TEXT PRIMARY KEY,
                content TEXT
            )
        ''')
        conn.commit()


def send_request():
    """Send request"""
    request_id = str(uuid.uuid4())  # Generate unique request ID
    # request_content = "Hello my world 2 from Server A"
    request_content = json.dumps({"message": 0.1, "other": 'my_love'})

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO requests (id, content) VALUES (?, ?)', (request_id, request_content))
        conn.commit()


    print(f"Server A sent request: {request_id} - {request_content}")
    return request_id


def wait_for_response(request_id):
    """Wait and receive response"""
    while True:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT content FROM responses WHERE id = ?', (request_id,))
            response = cursor.fetchone()

            if response:
                print(f"Server A received response: {response[0]}")
                # Remove processed response
                cursor.execute('DELETE FROM responses WHERE id = ?', (request_id,))
                conn.commit()
                break
        time.sleep(1)


def main():
    init_db()

    # Send request and wait for response
    request_id = send_request()
    wait_for_response(request_id)


if __name__ == "__main__":
    main()