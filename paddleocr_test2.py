from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')

image_path = 'testimg1.jpeg'
image = Image.open(image_path)
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

result = ocr.ocr(image_path, cls=True)

try:
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
except Exception as e:
    print("Error extracting OCR data:", e)
    boxes, txts, scores = [], [], []

font_path = 'Roboto-Regular.ttf'
im_show = draw_ocr(image_cv, boxes, txts, scores, font_path=font_path)
im_show_pil = Image.fromarray(cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB))
im_show_pil.save('annotated_image.jpg')

with open('ocr_results.txt', 'w') as file:
    for text in txts:
        file.write(f"{text}\n")
