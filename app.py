import os
from flask import Flask, render_template, request, redirect

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# Import datetime for timestamping
from datetime import datetime
from flask import send_from_directory
# Add a list to store history entries
history = []
app = Flask(__name__)

# Set environment variable to turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the Keras model
model = load_model('model.h5')

# Dictionary to map predictions to human-readable labels
dic = {0: 'LESS SWEET(หวานน้อย-หวานปานกลาง)', 1: 'NO SWEET(หวานน้อยมาก-ไม่หวาน)', 2: 'SWEET(หวาน-หวานมาก)'}

def image_processing_function(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class label
    predicted_class = dic[np.argmax(predictions)]

    return predicted_class

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Welcome to sweet project"

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        # Get the uploaded image
        img = request.files['my_image']

        # Save the image
        img_path = "static/" + img.filename
        img.save(img_path)

        # Perform image processing and get predictions
        prediction = image_processing_function(img_path)
        # Store the entry in the history list including file information
        entry = {
            'img_path': img_path,
            'prediction': prediction,
            'timestamp': datetime.now(),
            'filename': img.filename
        }
        history.append(entry)

        return render_template("index.html", prediction=prediction, img_path=img_path)

@app.route("/history", methods=['GET', 'POST'])
def show_history():
    if request.method == 'POST':
        # Get the feedback value from the form
        feedback_value = request.form.get('user_feedback')

        # Use the feedback value as needed, for example, store it in the history entry
        # Here, we assume that the feedback_value can be 'correct' or 'incorrect'
        # Modify this part according to your specific use case
        entry_index = int(request.form.get('entry_index'))  # Assuming you have a hidden input with entry_index
        if entry_index is not None and 0 <= entry_index < len(history):
            # Check if user_feedback is not already set
            if 'user_feedback' not in history[entry_index]:
                history[entry_index]['user_feedback'] = feedback_value

    return render_template("history.html", history=history)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
