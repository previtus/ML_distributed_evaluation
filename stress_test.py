# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
from timeit import default_timer as timer


PORT = "5000"
KERAS_REST_API_URL = "http://localhost:"+PORT+"/predict"
IMAGE_PATH = "small.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 100
SLEEP_COUNT = 0.05 # evaluation time grows

NUM_REQUESTS = 100
SLEEP_COUNT = 0.15 # here we are still making it

# will highly depend on network/io/... situation

def call_predict_endpoint(n):
	# load the input image and construct the payload for the request
	start = timer()

	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": image}

	# submit the request
	r = requests.post(KERAS_REST_API_URL, files=payload).json()

	end = timer()
	t = end - start

	# ensure the request was sucessful
	if r["success"]:
		print("[INFO] thread {} OK".format(n), t)

	# otherwise, the request failed
	else:
		print("[INFO] thread {} FAILED".format(n), t)

# loop over the number of threads
for i in range(0, NUM_REQUESTS):
	# start a new thread to call the API
	t = Thread(target=call_predict_endpoint, args=(i,))
	t.daemon = True
	t.start()
	time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
time.sleep(300)