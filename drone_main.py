from commander import Controller
from flask import Flask, request, jsonify

'''def main():
    # Instantiate the Controller
    controller = Controller()

    # Test NLP command
    command = "drone move left 100 and go down 40 and front 30"
    controller.commendor(command)  # This will process the command and print the results
'''
app = Flask(__name__)

# Variable to control the loop
should_continue = True

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



