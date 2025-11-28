from flask import Flask, request, jsonify
from inference import predict

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = predict(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
