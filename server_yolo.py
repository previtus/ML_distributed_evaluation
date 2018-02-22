# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model
from keras import backend as K

from threading import Thread
import time

import gc

import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import io
import os, sys
from timeit import default_timer as timer
from yolo_handler import run_on_image, run_on_single_crop

from yolo_handler import use_path_which_exists

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

app = flask.Flask(__name__)
darkflow_model = None
model = None
REUSE = False

def load_model_resnet():
    global model
    model = ResNet50(weights="imagenet")
    global graph
    graph = tf.get_default_graph()

def load_model_yolo(model_path):
    #global graph
    #graph = tf.get_default_graph()

    model_path = os.path.expanduser(model_path)
    print(model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

    global darkflow_model
    yolo_model = load_model(model_path)
    print('{} model, anchors, and classes loaded.'.format(model_path))

    global sess
    sess = K.get_session()

    global graph
    graph = tf.get_default_graph()

    global REUSE
    REUSE = False

def ReuseTrue():
    global REUSE
    REUSE = True

def prepare_image(image, target):
   # if the image mode is not RGB, convert it
   if image.mode != "RGB":
      image = image.convert("RGB")

   # resize the input image and preprocess it
   image = image.resize(target)
   image = img_to_array(image)
   image = np.expand_dims(image, axis=0)
   image = imagenet_utils.preprocess_input(image)

   # return the processed image
   return image

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


@app.route("/predict", methods=["POST"])
def predict():
   # Evaluate data
   data = {"success": False}

   if flask.request.method == "POST":
      if flask.request.files.get("image"):
         # read the image in PIL format
         image = flask.request.files["image"].read()
         image = Image.open(io.BytesIO(image))

         image = prepare_image(image, target=(224, 224))

         with graph.as_default():
            # evaluate image
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            for (imagenetID, label, prob) in results[0]:
               r = {"label": label, "probability": float(prob)}
               data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

   # return the data dictionary as a JSON response
   return flask.jsonify(data)

@app.route("/yolo_full", methods=["POST"])
def yolo_full():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            with graph.as_default():
                # evaluate image
                bboxes = run_on_image(image, darkflow_model, sess) # aka many crops

                data["bboxes"] = bboxes

                # indicate that the request was a success
                data["success"] = True

    return flask.jsonify(data)


@app.route("/yolo_single_crop", methods=["POST"])
def yolo_single_crop():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            with tf.variable_scope("model", reuse=REUSE):

                with graph.as_default():
                    # evaluate image
                    bboxes = run_on_single_crop(image, darkflow_model, sess)

                    data["bboxes"] = bboxes

                    # indicate that the request was a success
                    data["success"] = True

            ReuseTrue()
            gc.collect()

    return flask.jsonify(data)

def mem_monitor_deamon():
    import resource
    import subprocess
    while (True):
        """
        #import psutil
        #process = psutil.Process(os.getpid())
        #mem = process.get_memory_info()[0] / float(2 ** 20)
        #return mem

        rusage_denom = 1024.
        if sys.platform == 'darwin':
            # ... it seems that in OSX the output is different units ...
            rusage_denom = rusage_denom * rusage_denom
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
        #return mem
        """

        out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                               stdout=subprocess.PIPE).communicate()[0].split(b'\n')
        vsz_index = out[0].split().index(b'RSS')
        mem = float(out[1].split()[vsz_index]) / 1024

        print("Memory:", mem)
        time.sleep(2.0) # check every 2 sec


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    #load_model_resnet()

    yolo_paths = ["/home/ekmek/YAD2K/", "/home/vruzicka/storage_pylon2/YAD2K/"]
    path_to_yolo = use_path_which_exists(yolo_paths)
    model_h5 = 'yolo.h5'
    yolo_path = path_to_yolo + 'model_data/' + model_h5

    load_model_yolo(yolo_path)

    t = Thread(target=mem_monitor_deamon, args=())
    t.daemon = True
    t.start()


    app.run()
    # On server:
    #app.run(host='0.0.0.0', port=8123)
