import time

class Practice:
    def __init__(self, name):
        self.name = name  # Instance variable to store the name
    
    def greet(self):
        global should_continue  # Access the global variable to control the loop
        while should_continue:  # Keep printing "Hello" until state is False
            print("Hello")
            time.sleep(1)

    def stop(self):
        global should_continue
        should_continue = False  # Stop the loop by setting the flag to False
from flask import Flask, request, jsonify
import threading
import time

# Global variable to control the loop
should_continue = True

app = Flask(__name__)

@app.route('/send_text', methods=['POST'])
def receive_text():
    global should_continue  # Access the global variable

    # Get JSON payload
    data = request.get_json()  # Get JSON body from the request
    if not data or 'text' not in data:
        return jsonify({"error": "Text not provided"}), 400

    text = data['text']
    print(f"Received text: {text}")

    # If the text is 'stop', set the flag to False to stop the loop
    if text.lower() == 'stop':
        should_continue = False

    return jsonify({"message": "Text received successfully"}), 200

def background_task():
    global should_continue
    # Instantiate Practice class
    person = Practice("John")
    person.greet()  # Start the greet loop

if __name__ == '__main__':
    # Start the background task in a separate thread
    threading.Thread(target=background_task, daemon=True).start()

    # Run Flask server
    app.run(host='10.251.31.128', port=5000)
