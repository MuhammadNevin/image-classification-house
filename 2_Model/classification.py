import cv2    
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = (150, 150)
FOLDER = 'training_9'

def create_model(x, y, z):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (x, y, z)),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

    return model

def prediction(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # wh = min(image.shape[0], image.shape[1]) - 1
    # image = get_random_crop(image, wh, wh)
    image = cv2.resize(image, IMAGE_SIZE) 
    image = np.array(image, dtype = 'float32')
    image = image / 255.0

    # print(np.array([image]))
    model_vgg = VGG16(weights='imagenet', include_top=False)
    image_features = model_vgg.predict(np.array([image]))
    n_train, x, y, z = image_features.shape

    model2_vgg_load = create_model(x, y, z)
    model2_vgg_load.load_weights(FOLDER + '/model.h5')

    predictions_vgg = model2_vgg_load.predict(image_features)     # Vector of probabilities
    pred_labels_vgg = ((predictions_vgg > 0.45)+0).ravel() # We take the highest probability
    
    if(pred_labels_vgg):
        np.set_printoptions(suppress=True)
        print("prediction: tidak layak (" + str(predictions_vgg[0][0]) + ")")
    else :
        np.set_printoptions(suppress=True)
        print("prediction: layak (" + str(predictions_vgg[0][0]) + ")")
    
prediction('testing/tidak-layak/2702.jpg')