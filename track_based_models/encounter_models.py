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

from .base_model import hybrid_pool_layer_2, Normalizer
from .dual_track_model import DualTrackModel
from .util import minute, lin_interp, cos_deg, sin_deg 

assert K.image_data_format() == 'channels_last'



class ConvNetModel4(DualTrackModel):
    
    delta = 10 * minute
    time_points = 72
    window = time_points * delta
    time_point_delta = 1000000000000

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_true_vals = [1]
    data_false_vals = [2, 3]
    data_defined_vals = [1, 2, 3]
    data_undefined_vals = [0]

    data_far_time = 3 * 10 * minute
    # time_points = window // delta
    base_filter_count = 32

    fc_nodes = 128
    
    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 14))
        y = input_layer
         
        input_layer = Input(shape=(self.time_points, 14))
        y = input_layer
        y = Conv1D(depth, 6)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        y = Flatten()(y)
        
        y = Dense(self.fc_nodes)(y)
        y = ELU()(y)

        y = Dense(1)(y)
        y = Activation('sigmoid')(y)

        # (1 10) (10 28) (28 64)

        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer)
        # opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.05)
        opt = optimizers.Nadam(lr=0.005, schedule_decay=0.5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  
    
    def preprocess(self, x):
        return self.normalizer.norm(x)
    
    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, sample_weight=sample_weight[:, 0],
                      validation_split=validation_split, validation_data=validation_data)
        return self

    def predict(self, x):
        x1 = self.preprocess(x)
        return self.model.predict(x1) > 0.5
    
class ConvNetModel5(ConvNetModel4):
    
    window = 73 * ConvNetModel4.delta

    def preprocess(self, x):
        x0 = np.asarray(x) # 3 / 4
        x = 0.5 * (x0[:, 1:, :] + x0[:, :-1, :])
        dxy = x0[:, 1:, 3:5] - x0[:, :-1, 3:5]
        x[:, :, 3:5] = dxy
        dxy = x0[:, 1:, 9:11] - x0[:, :-1, 9:11]
        x[:, :, 9:11] = dxy
        return x

