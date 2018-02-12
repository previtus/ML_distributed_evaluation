# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import io
from timeit import default_timer as timer

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

app = flask.Flask(__name__)
model = None

def load_model():
	global model
	model = ResNet50(weights="imagenet")
	global graph
	graph = tf.get_default_graph()

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

@app.route("/time", methods=["POST"])
def time():
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


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()
	# On server:
	#app.run(host='0.0.0.0', port=8123)