# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# TODO: Make all other necessary imports.

import warnings
warnings.filterwarnings('ignore')

import argparse

import numpy as np
import matplotlib.pyplot as plt
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
tfds.disable_progress_bar()

import json 
import tensorflow_hub as hub
import time

from PIL import Image

#Process

#Process Images

# TODO: Create the process_image function
def process_image(image):
    
    image = tf.cast(image, tf.float32)
    image =  tf.image.resize(image, ( 224,  224))
    image /= 255
    
    return(image)


def predict(image_path, model,top_k):
    
    im = Image.open(image_path)
    test_image = np.asarray(im)

    image = process_image(test_image)
    
    image = np.expand_dims(image, axis=0)
    
    ps = model.predict(image)
    top_k=int(top_k)
    
    if top_k>0 and top_k<=len(ps[0]):
        probs, classes= tf.nn.top_k(ps, k=top_k)

        probs = probs.numpy()[0]
        classes = classes.numpy()[0]
       

    return(probs,classes)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    
    image_path = args.arg1
    
    saved_keras_model_filepath = args.arg2
    
    map_json= args.category_names
    
    top_k = args.top_k
    
    print('image_path', image_path)
    
    print('saved_keras_model_filepath', saved_keras_model_filepath)
    
    
    if map_json == None:
        print("No flower labels")
    else:
        print("flower labels file:", map_json)
        
    if top_k == None:
        print("No most likely classes selected, default = 1")
        top_k=1

    else:
        print("Return the top ", top_k ,"most likely classes");
        
    reloaded_keras_model = tf.keras.models.load_model(saved_keras_model_filepath,custom_objects={'KerasLayer':hub.KerasLayer},compile=False)

    #reloaded_keras_model.summary()
    
    probs,classes=predict(image_path, reloaded_keras_model,top_k)
    
    print('probs size',probs)
       
    for i in range(len(probs)):
        
        if map_json!=None:
            
            with open(map_json, 'r') as f:
                class_names = json.load(f)
            
            print(i+1,"Class",class_names[str(classes[i]+1)],", prob:", round(probs[i]*100,3),"%")
            
        else:
            print(i+1,"Class number",classes[i]+1,", prob:",round(probs[i]*100,3),"%")
    
    print("DONE!")
    
    # python predict.py ./test_images/cautleya_spicata.jpg Model_1622846207.h5
    # python predict.py ./test_images/cautleya_spicata.jpg Model_1622846207.h5 --category_names label_map.json
    # python predict.py ./test_images/cautleya_spicata.jpg Model_1622846207.h5 --top_k 3 --category_names label_map.json
    
    
    
    
