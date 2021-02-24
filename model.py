from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialMaskModel(object):    

    def __init__(self, model_weights_file):
        # load weights into the new model
        self.loaded_model = load_model(model_weights_file)
        
    def predict_mask(self, img):
        global session
        set_session(session)
        preds = self.loaded_model.predict(img)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # return "With Mask" if mask > withoutMask else "Without Mask"
        return label
