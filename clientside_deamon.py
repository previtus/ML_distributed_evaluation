# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
from timeit import default_timer as timer
import numpy

PORT = "9999"
YOLO_KERAS_REST_API_URL = "http://localhost:"+PORT+"/get_all_bboxes"
SLEEP_COUNT = 1.0
SLEEP_COUNT = 0.01 #server

def check_for_updates():
    times = []
    k = 50

    while True:
        try:
            r = requests.post(YOLO_KERAS_REST_API_URL).json()
        except Exception:
            #print("server not setup yet, passing...")
            pass
        else:
            bboxes = r["bboxes"]

            if len(bboxes) > 0:
                #print("got ",len(bboxes), "bboxes:", bboxes)
                for bbox in bboxes:
                    # bbox is in [uid, timestamp, bboxarray]
                    time_difference = timer() - bbox[1]
                    times.append(time_difference)
                    uid = bbox[0]
                    print("[INFO] thread/bbox uid {} ".format(uid), time_difference, k)
                    k = k - 1

                if k <= 0:
                    print("Time average over last 50+batch:", numpy.mean(times))
                    k = 50
                    times = []

        time.sleep(SLEEP_COUNT)

t = Thread(target=check_for_updates, args=())
t.daemon = True
t.start()
t.join()