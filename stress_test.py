# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
from timeit import default_timer as timer


PORT = "5000"
YOLO_KERAS_REST_API_URL = "http://localhost:"+PORT+"/yolo_image"
IMAGE_PATH = "small.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests

NUM_REQUESTS = 1000 # server
SLEEP_COUNT = 0.05

NUM_REQUESTS = 50 # local
SLEEP_COUNT = 1.0

# will highly depend on network/io/... situation

def call_predict_endpoint(n, q):
	# load the input image and construct the payload for the request
	start = timer()

	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": image}

	# submit the request
	r = requests.post(YOLO_KERAS_REST_API_URL, files=payload).json()

	end = timer()
	t = end - start

	# ensure the request was sucessful
	if r["success"]:
		print("[INFO] thread {} OK".format(n), t)
		q.put({n: r})

	# otherwise, the request failed
	else:
		print("[INFO] thread {} FAILED".format(n), t)

import queue
q = queue.Queue()
threads = []

for i in range(0, NUM_REQUESTS):
	# start a new thread to call the API
	t = Thread(target=call_predict_endpoint, args=(i,q))
	t.daemon = True
	t.start()
	threads.append(t)
	time.sleep(SLEEP_COUNT)

for t in threads:
	t.join()

for item in iter(q.get, None):
	print(item)

# insert a long sleep so we can wait until the server is finished
# processing the images
#time.sleep(300)