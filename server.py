# -*- coding:utf-8 -*-
# @time :2019.09.24
# @IDE : pycharm
# @autor :lxztju
# @github : https://github.com/lxztju

import io

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import flask
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




label_id_name_dict = \
    {
        "0": "狗",
        "1": "猫"
    }

app = Flask(__name__)
# 首先将model定义为None，原因在后面解释。
model = None

pb_path = "./"
input_size = 300

def load_model():
    # load the pre-trained Keras model (this model is saved by tf.saved_model.builder.SavedModelBuilder
    # which contains a variable folder and a pb file.
    
    global model
    model =  tf.saved_model.loader.load(sess, [tag_constants.SERVING], pb_path)

    global graph
    graph = tf.get_default_graph()
    
    

def center_img(img, size=None, fill_value=255):
    """
    center img in a square background
    """
    h, w = img.shape[:2]
    if size is None:
        size = max(h, w)
    shape = (size, size) + img.shape[2:]
    background = np.full(shape, fill_value, np.uint8)
    center_x = (size - w) // 2
    center_y = (size - h) // 2
    background[center_y:center_y + h, center_x:center_x + w] = img
    return background

def preprocess_img(img):
    """
    image preprocessing
    you can add your special preprocess method here
    """
    resize_scale = input_size / max(img.size[:2])
    img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]
    img = center_img(img,input_size)
    img = img[np.newaxis, :, :, :] 
    img = img.astype(np.float32) / 225
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img[..., 0] /= std[0]
    img[..., 1] /= std[1]
    img[..., 2] /= std[2]
    return img




@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    print(data)

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print("Hello")
        if flask.request.files.get("image"):
            print("world")
        
            # read the image in PIL format
            
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            img = preprocess_img(image)



            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                signature_key = 'predict_images'

				# get signature
                signature = model.signature_def
                # get tensor name
                in_tensor_name = signature[signature_key].inputs['input_img'].name
                out_tensor_name = signature[signature_key].outputs['output_score'].name
                # get tensor
                input_images = sess.graph.get_tensor_by_name(in_tensor_name)
                output_score = sess.graph.get_tensor_by_name(out_tensor_name)
                # run
                pred_score = sess.run([output_score], feed_dict={input_images: img})
                pred_label = np.argmax(pred_score[0], axis=1)[0]

            data["predictions"] = []
            data["predictions"].append(label_id_name_dict[str(pred_label)])

            # indicate that the request was a success
            data["success"] = True
            print(data["success"])

    # return the data dictionary as a JSON response
    return jsonify(data)
    
    
    
# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    sess = tf.Session()
    load_model()
    app.run(host='0.0.0.0', port =5000 )


