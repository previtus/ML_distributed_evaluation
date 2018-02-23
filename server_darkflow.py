# import the necessary packages
from darkflow_handler import load_model, run_on_image, run_on_images
from keras.preprocessing.image import img_to_array

from threading import Thread
import time

from PIL import Image
import flask
import io
import os
from timeit import default_timer as timer
import serverside_queues

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

app = flask.Flask(__name__)
darkflow_model = None
REUSE = False

def load_model_darkflow():
    global darkflow_model
    darkflow_model = load_model()
    print('Model loaded.')

@app.route("/time_transfer", methods=["POST"])
def time_transfer():
   # Time transferring and loading the file

   data = {"success": False}
   start = timer()

   if flask.request.method == "POST":
      if flask.request.files.get("image"):
         image = flask.request.files["image"].read()
         image = Image.open(io.BytesIO(image))

         data["imageshape"] = image.size

         end = timer()
         data["internal_time"] = end - start
         data["success"] = True


   # return the data dictionary as a JSON response
   return flask.jsonify(data)


@app.route("/yolo_image", methods=["POST"])
def yolo_image():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = img_to_array(image)

            # evaluate image
            bboxes = run_on_image(image, darkflow_model)
            data["bboxes"] = bboxes

            # indicate that the request was a success
            data["success"] = True

    return flask.jsonify(data)

@app.route("/enqueue_image", methods=["POST"])
def enqueue_image():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            uid = flask.request.files["uid"].read()

            uid = int(uid)

            image = Image.open(io.BytesIO(image))
            image = img_to_array(image)

            serverside_queues.put_crop_to_queue(image, uid)
            # indicate that the request was a success
            data["success"] = True

    return flask.jsonify(data)

@app.route("/get_all_bboxes", methods=["POST"])
def get_all_bboxes():
    data = {"success": False}
    if flask.request.method == "POST":
        bbox_objects = serverside_queues.get_all_bboxes_from_queue()
        #print(bbox_objects)

        data["bboxes"] = bbox_objects
        data["success"] = True

    return flask.jsonify(data)


def mem_monitor_deamon():
    import subprocess
    while (True):
        out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                               stdout=subprocess.PIPE).communicate()[0].split(b'\n')
        vsz_index = out[0].split().index(b'RSS')
        mem = float(out[1].split()[vsz_index]) / 1024

        print("Memory:", mem)
        time.sleep(2.0) # check every 2 sec

def queue_evaluator_deamon(n, wait):
    # job of this deamon is to be always checking the crops queue and evaluating it

    # test with just getting the data from it at start...
    while (True):
        crops = serverside_queues.get_crops_from_queue(n)

        if len(crops) > 0:
            #crops = serverside_queues.get_all_from_queue()

            uids_objects = [items[0] for items in crops]
            times_objects = [items[1] for items in crops]
            image_objects = [items[2] for items in crops] # get images

            print("Loaded",len(crops),"crops with uids",uids_objects,"as a batch. (... will evaluate)")

            results_bboxes = run_on_images(image_objects=image_objects, model=darkflow_model)

            print("Evaluated uids", uids_objects, "crops as a batch.")

            serverside_queues.put_bboxes_to_queue(results_bboxes, uids_objects, times_objects)

        if wait > 0.0:
            time.sleep(wait)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model_darkflow()

    t = Thread(target=mem_monitor_deamon, args=())
    t.daemon = True
    t.start()

    n = 2
    wait = 0.0
    t = Thread(target=queue_evaluator_deamon, args=(n, wait))
    t.daemon = True
    t.start()

    app.run()
    # On server:
    #app.run(host='0.0.0.0', port=8123)
