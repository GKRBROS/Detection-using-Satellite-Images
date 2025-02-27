from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
import os
import io
from matplotlib import pyplot as plt
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

def calculate_rgb_difference(image_before, image_after, threshold_value):
    diff = np.abs(image_after.astype(np.float32) - image_before.astype(np.float32))
    rgb_difference = np.mean(diff, axis=2)
    changes_detected = np.where(rgb_difference > threshold_value, 1, 0)
    return changes_detected

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image_before' not in request.files or 'image_after' not in request.files:
        return "Please upload both images.", 400

    image_before_file = request.files['image_before']
    image_after_file = request.files['image_after']
    threshold_value = float(request.form.get('threshold', 20))

    image_before_path = os.path.join(UPLOAD_FOLDER, 'before.jpg')
    image_after_path = os.path.join(UPLOAD_FOLDER, 'after.jpg')
    image_before_file.save(image_before_path)
    image_after_file.save(image_after_path)

    image_before = load_image(image_before_path)
    image_after = load_image(image_after_path)

    if image_before.shape != image_after.shape:
        image_after = resize_image(image_after, image_before.shape[:2])

    changes_detected = calculate_rgb_difference(image_before, image_after, threshold_value)

    plt.figure(figsize=(6, 3))
    plt.imshow(changes_detected, cmap='gray')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
