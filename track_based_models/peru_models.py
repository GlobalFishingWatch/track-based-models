from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, ELU, ReLU, Input, Conv1D
from keras.layers import BatchNormalization, MaxPooling1D, Concatenate
from keras.layers import Cropping1D, AveragePooling1D
from keras.layers.core import Activation, Reshape
from keras import optimizers
from .util import hour, minute
from .base_model import hybrid_pool_layer_2, Normalizer, hybrid_pool_layer

from .single_track_model import SingleTrackModel
from . import util
from .util import minute, lin_interp, cos_deg, sin_deg 


from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects



class ConvNetModel5(SingleTrackModel):
    
    delta = 15 * minute
    window = (29 + 4*6) *  delta
    time_points = window // delta

    base_filter_count = 8
    fc_nodes = 32

    data_source_lbl='fishing' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute

    time_point_delta = 10000000000 # None, breaks downstream

    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(self.time_points, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Dropout(0.3)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer(y)
        y = Dropout(0.4)(y)

        depth = 2 * depth 
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Dropout(0.4)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer(y)
        y = Dropout(0.5)(y)

        y = Conv1D(self.fc_nodes, 10, activation=None)(y)
        # 1 - 2 - 6 - 12 - 17 + 4 * (n - 1)

        # y = Flatten()(y)
        # y = Dense(self.fc_nodes)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)

        # y = Dense(self.fc_nodes)(y)
        y = Conv1D(self.fc_nodes, 1, activation=None)(y)
        y = ELU()(y)
        y = Dropout(0.5)(y)

        y = Conv1D(1, 1, activation=None)(y)
        y = Activation('sigmoid')(y)
        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        #opt = optimizers.Adam(lr=0.01, decay=0.5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"],
            sample_weight_mode="temporal")
        self.model = model 

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[]):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      verbose=verbose, callbacks=callbacks)
