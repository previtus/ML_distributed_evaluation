import requests
from timeit import default_timer as timer

PORT = "5000"
YOLO_KERAS_REST_API_URL = "http://localhost:"+PORT+"/enqueue_image"

IMAGE_PATH = "small.jpg"
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image, "uid": "42", "time":str(timer()) }

# submit the request
start = timer()
r = requests.post(YOLO_KERAS_REST_API_URL, files=payload).json()
end = timer()
print("Time", (end-start))
print("request data", r)
