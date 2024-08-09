import requests
import base64
import json

# image encoded in base64
image = str(input("Enter image directory: "))
with open(image, 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

payload = {'image': base64_image}

# contacting API
response = requests.post('http://localhost:5000/ocr', json=payload)

# check if the request was successful
if response.status_code == 200:
    # receiving results in json file
    result = response.json()

    # saving json file
    with open('ocr_results.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print("OCR results saved to 'ocr_results.json'")
else:
    print(f"Error: {response.status_code}")