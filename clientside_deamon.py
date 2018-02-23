# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
from timeit import default_timer as timer


PORT = "5000"
YOLO_KERAS_REST_API_URL = "http://localhost:"+PORT+"/get_all_bboxes"
SLEEP_COUNT = 0.2

def check_for_updates():
    while True:
        try:
            start = timer()
            r = requests.post(YOLO_KERAS_REST_API_URL).json()
        except Exception:
            #print("server not setup yet, passing...")
            pass
        else:

            end = timer()
            #print("Request time", (end - start))
            #print(r)

            bboxes = r["bboxes"]

            if len(bboxes) > 0:
                #print("got ",len(bboxes), "bboxes:", bboxes)
                for bbox in bboxes:
                    # bbox is in [uid, timestamp, bboxarray]
                    time_difference = timer() - bbox[1]
                    uid = bbox[0]
                    print("[INFO] thread/bbox uid {} ".format(uid), time_difference)

        time.sleep(SLEEP_COUNT)

t = Thread(target=check_for_updates, args=())
t.daemon = True
t.start()
t.join()