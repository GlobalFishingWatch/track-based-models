from __future__ import division
from __future__ import print_function
import numpy as np
import os
import keras
from keras.models import Sequential, Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, MaxoutDense, LeakyReLU, ELU, Input
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.core import Activation
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers
from keras import regularizers
from keras.models import load_model

from .dual_track_model import DualTrackModel
from .util import minute, lin_interp, cos_deg, sin_deg 

assert K.image_data_format() == 'channels_last'


class Normalizer(object):
    
    def fit(self, features):
        features = np.asarray(features)
        self.mean = features.mean(axis=(0, 1), keepdims=True)
        self.std = features.std(axis=(0, 1), keepdims=True)
        return self
        
    def norm(self, features):
        features = np.asarray(features)
        return (features - self.mean) / self.std
    
    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)
        
    @classmethod
    def load(cls, path):
        archive = np.load(path)
        obj = cls()
        obj.mean = archive['mean']
        obj.std = archive['std']
        return obj


def hybrid_pool_layer(x, pool_size=2):
    return Conv1D(int(x.shape[-1]), 1)(
        keras.layers.concatenate([
            MaxPooling1D(pool_size, strides=2)(x),
            AveragePooling1D(pool_size, strides=2)(x)]))

def hybrid_pool_layer_2(x):
    depth = int(x.shape[-1])
    x2 = Conv1D(depth, 3, strides=2)(x)
    x2 = ELU()(x2)
    x2 = keras.layers.BatchNormalization(scale=False, center=False)(x2)
    return Conv1D(depth, 1)(keras.layers.concatenate([
                                      MaxPooling1D(3, strides=2)(x), x2]))    
    

# class BaseModel(object):
    
#     def flatten(self, x):
#         x = np.asarray(x)
#         return x.reshape(x.shape[0], -1)
    
#     def save(self, path):
#         self.model.save(os.path.join(path, 'model.h5'))
#         self.normalizer.save(os.path.join(path, 'norm.npz'))
        
#     @classmethod
#     def load(cls, path):
#         mdl = cls()
#         mdl.model = load_model(os.path.join(path, 'model.h5'))
#         mdl.normalizer = Normalizer.load(os.path.join(path, 'norm.npz'))
#         return mdl


class ConvNetModel4(DualTrackModel):
    
    delta = 10 * minute
    time_points = 73
    window = time_points * delta
    time_point_delta = 8

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_true_vals = [1]
    data_false_vals = [2, 3]
    data_far_time = 3 * 10 * minute
    # time_points = window // delta
    base_filter_count = 32
    fc_nodes = 128
    
    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 9))
        y = input_layer
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        
        y = Conv1D(depth, 5, dilation_rate=2)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5, dilation_rate=2)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        
        y = Conv1D(depth, 5, dilation_rate=3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5, dilation_rate=3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        
        y = Conv1D(depth, 5, dilation_rate=2)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5, dilation_rate=2)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)

        y = Conv1D(depth, 5, dilation_rate=1)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5, dilation_rate=1)(y)
        y = ELU()(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        # (1 9) (9 25) (25 49) (49 65) (65 73)

        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        #opt = optimizers.Adam(lr=0.01, decay=0.5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"],
            sample_weight_mode="temporal")
        self.model = model  
    
    def preprocess(self, x):
        return self.normalizer.norm(x)
    
    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), -1, 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, sample_weight=sample_weight,
                      validation_split=validation_split, validation_data=validation_data)
        return self

    def predict(self, x):
        x1 = self.preprocess(x)
        return self.model.predict(x1)[:, :, 0] > 0.5
    
# class ConvNetModel5(ConvNetModel4):
    
#     delta = 10 * minute
#     time_points = 71 
#     window = (time_points + 1) * delta
#     time_point_delta = 1

#     data_source_lbl='transshiping' 
#     data_target_lbl='is_target_encounter'
#     data_true_vals = [1]
#     data_false_vals = [2, 3]
#     data_far_time = 3 * 10 * minute

#     def preprocess(self, x):
#         x = np.asarray(x) # 3 / 4
#         dxy = x[:, 1:, 3:5] - x[:, :-1, 3:5]
#         x = 0.5 * (x[:, 1:, :] + x[:, :-1, :])
#         x[:, :, 3:5] = dxy
#         return x

