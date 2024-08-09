import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import io

# initializing paddleocr and flask
ocr = PaddleOCR(use_angle_cls=True, lang='en')
app = Flask(__name__)


# decodes received base64 into image
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data))
    return image


@app.route('/ocr', methods=['POST'])
def ocr_api():
    try:
        # receives base64
        data = request.json
        base64_image = data.get('image')

        # decodes base64 into image
        image = base64_to_image(base64_image)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # runs image through ocr
        result = ocr.ocr(image_cv, cls=True)

        # extracts all the text
        txts = [line[1][0] for line in result[0]]

        # returns text results in json file
        return jsonify({'recognized_texts': txts})

    except Exception as e:
        # if error has occurred, returns 500
        return jsonify({'error': str(e)}), 500


# running and hosting the api
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
