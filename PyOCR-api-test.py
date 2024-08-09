import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import io

# Initialize PaddleOCR and Flask
ocr = PaddleOCR(use_angle_cls=True, lang='en')
app = Flask(__name__)

# Decodes received base64 into image
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data))
    return image

@app.route('/ocr', methods=['POST'])
def ocr_api():
    try:
        # Receive base64 image
        data = request.json
        base64_image = data.get('image')

        # Decode base64 into image
        image = base64_to_image(base64_image)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Check if the image has been correctly converted
        if image_cv is None or image_cv.size == 0:
            return jsonify({'error': 'Image conversion failed'}), 400

        # Run OCR on the image
        result = ocr.ocr(image_cv, cls=True)

        # Extract all the text
        txts = [line[1][0] for line in result[0]]

        # Return text results in JSON format
        return jsonify({'recognized_texts': txts})

    except Exception as e:
        # If an error has occurred, return 500
        return jsonify({'error': str(e)}), 500

# Running and hosting the API
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)