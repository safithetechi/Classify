

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile
from shutil import copy

from flask import Flask,render_template, request,redirect, url_for,jsonify
from werkzeug import secure_filename
import _thread
from threading import Thread
import caffe

import numpy as np
import pickle

from six.moves import urllib
import tensorflow as tf

import string
import random


FLAGS = None

results1=['', '', '','','']
results2=['', '', '','','']

tmpimage=""


app =Flask(__name__)


UPLOAD_FOLDER='uploads'

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])

parser = argparse.ArgumentParser()

InceptionPath ='inception-2015-12-05'
PlacesPath = 'models_places'



parser.add_argument(
      '--model_dir',
      type=str,
      default=InceptionPath,
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )

parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )

parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )

FLAGS, unparsed = parser.parse_known_args()

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  global results1

  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()


  create_graph()

  with tf.Session() as sess:

    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)


    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    count=0
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      results1[count]=human_string+": "+str(score)

    print(results1)
    return results1



def main(_):
    global tmpimage
    image = (FLAGS.image_file if FLAGS.image_file else
    os.path.join(FLAGS.model_dir, tmpimage))
    run_inference_on_image(image)
    os.remove('inception-2015-12-05/'+tmpimage)
def start():
    result=[]
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    print("App started")




def classify_scene(fpath_design, fpath_weights, fpath_labels, im):

	global results2
	# initialize net
	net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)

	# load input and configure preprocessing
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1)) # TODO - remove hardcoded path
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

	# since we classify only one image, we change batch size from 10 to 1
	net.blobs['data'].reshape(1,3,227,227)

	# load the image in the data layer
	net.blobs['data'].data[...] = transformer.preprocess('data', im)

	# compute
	out = net.forward()

	# print top 5 predictions - TODO return as bytearray?
	with open(fpath_labels, 'rb') as f:

		labels = pickle.load(f)
		top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]

		for i, k in enumerate(top_k):
			results2[i]=labels[k]


	print(results2)
	return results2




def start_model2(image):

	# fetch pretrained models
	fpath_design = PlacesPath+'/deploy_alexnet_places365.prototxt'
	fpath_weights = PlacesPath+'/alexnet_places365.caffemodel'
	fpath_labels = 'resources/labels.pkl'

	# fetch image
	im = caffe.io.load_image(image)

	# predict
	return classify_scene(fpath_design, fpath_weights, fpath_labels, im)




def CountImages():
    files = os.listdir(UPLOAD_FOLDER)
    number_files = len(files)
    return number_files



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/upload',methods=['POST'])
def UploadFile():
    global tmpimage

    if 'file' not in request.files:
        return jsonify({"success":False})
    File = request.files['file']
    if File.filename=='':
        flash('No selected file')
        return jsonify({"success":False})

    if File and allowed_file(File.filename):
        Extention =File.filename.rsplit('.', 1)[1].lower()
        name=id_generator()+'.'+Extention
        File.filename=name
        PathForUploads=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(File.filename))
        tmpimage=name
        File.save(PathForUploads)
        copy(PathForUploads, 'inception-2015-12-05')
        return jsonify({"success":True})

    else:
        return jsonify({"success":False})

@app.route('/my_endpoint', methods=['GET','POST'])
def my_endpoint_handler():

    request=()
    test=False
    def handle_sub_view():
        with app.test_request_context():
            start()
    thread=Thread(target=handle_sub_view)
    thread.start()
    thread.join()
    return jsonify(results1)


@app.route('/places', methods=['GET','POST'])
def handler():
    global tmpimage
    def handle_sub_view():
        with app.test_request_context():
            start_model2('uploads/'+tmpimage)
    thread=Thread(target=handle_sub_view)
    thread.start()
    thread.join()
    return jsonify(results2)

def start():

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    print("App started")






if __name__ == '__main__':
    app.run(threaded=True)
