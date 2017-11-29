from __future__ import print_function
import keras
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

        
class AUCHistory(keras.callbacks.Callback):
    def __init__(self, input_len=1, *args, **kwargs):
        self.input_len = input_len
        super(AUCHistory, self).__init__(*args, **kwargs)
 
    def on_epoch_end(self, epoch, logs={}):
        if self.input_len == 1:
            y_pred = self.model.predict(self.model.validation_data[0])
        else:
            y_pred = self.model.predict(self.model.validation_data[:self.input_len])
        
        auc = roc_auc_score(self.model.validation_data[self.input_len], y_pred)
        print("\nEpoch validation AUC: {}\n".format(auc))

