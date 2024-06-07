import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    penoken = pickle.load(f)

# Load the TensorFlow Lite model
int_convo = tf.lite.Interpreter(model_path="convo.tflite")
int_convo.allocate_tensors()

input_details = int_convo.get_input_details()
output_details = int_convo.get_output_details()

# Optional: Print input and output details
print(input_details, output_details)

@app.route('/classification', methods=['POST'])
def classify():
    data = request.json
    text = data.get("text", "")
    
    seq = penoken.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=200, padding="post", truncating='post')
    pad = tf.cast(pad, tf.float32)

    in_index = input_details[0]["index"]
    int_convo.set_tensor(in_index, pad)

    int_convo.invoke()

    out_index = output_details[0]["index"]
    output_data = int_convo.get_tensor(out_index)
    
    # Return the first value of the output data
    return jsonify({"result": output_data[0].tolist()})

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
