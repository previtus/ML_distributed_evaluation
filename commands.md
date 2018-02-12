Commands to run the server and client and to connect to it
///////////////////////////////////////////////////////
Going from a testcase of connecting to distant *Jupyter notebook*:
  1.) on server
  jupyter notebook --no-browser --port=8889

  2.) on local
  ssh -N -f -L localhost:8888:localhost:8889 USERNAME@HOST
  (open tunnel, run in bg)

  open web site
  localhost:8888

We can use the same for our python server *server.py*:
Lets assume that we have started a interactive shell on our server, and so we need two tunnels (one from local to server and one from the server to the interactivelly run machine)
[on server]

interact --gpu --egress
#Load your Anacondas and other snakes ~~~ this is PSC Bridge specific:
#module load keras/2.0.6_anaconda # PSC Bridge specific,
#source activate $KERAS_ENV

python server.py
# tells me "Running on http://0.0.0.0:8123/"
# because of line 113: app.run(host='0.0.0.0', port=8123)

# to help see what is the name of the interactive machine call:
squeue -u USERNAME
# lets assume its name is bobmarley
# we can now connect to it via: ssh bobmarley

# lets build some tunnels then...
[on local]
[local 9999 >> server to bobmarley -> our desired gpu machine 8123]
ssh -N -f -L  9999:bobmarley:8123 USERNAME@HOST

You are there! Try with:
web address: http://localhost:9999/predict # should give you "Method Not Allowed"
curl -X POST -F image=@dog.jpg 'http://localhost:9999/predict' # imagine a dog.jpg here
or ultimatelly run the client:
  python client.py

"""
Time 0.07783991500036791
request data {'imageshape': [608, 608], 'internal_time': 0.007249436996062286, 'success': True}
Time 0.07160759600083111
request data {'imageshape': [608, 608], 'internal_time': 0.003656726999906823, 'success': True}
Time 0.1039813960014726
request data {'predictions': [{'label': 'cinema', 'probability': 0.3615420460700989}, {'label': 'unicycle', 'probability': 0.22204479575157166}, {'label': 'street_sign', 'probability': 0.05314052477478981}, {'label': 'jinrikisha', 'probability': 0.04694230109453201}, {'label': 'traffic_light', 'probability': 0.04335896298289299}], 'success': True}
"""