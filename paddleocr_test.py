from paddleocr import PaddleOCR, draw_ocr
import cv2
from PIL import Image
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')

image_path = 'testimg1.jpeg'
image = Image.open(image_path)

image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

result = ocr.ocr(image_path, cls=True)

for line in result[0]:
    text = line[1][0]
    print(text)

boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

im_show = draw_ocr(image_cv, boxes, txts, scores, font_path='Roboto-Regular.ttf')
im_show = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
im_show = Image.fromarray(im_show)
im_show.show()

