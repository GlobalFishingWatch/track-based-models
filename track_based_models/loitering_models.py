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
from keras.layers import Cropping1D, AveragePooling1D, Cropping1D, Conv2DTranspose
from keras.layers.core import Activation, Reshape
from keras import optimizers
import logging
from .util import hour, minute
from .base_model import hybrid_pool_layer_2, Normalizer
from .single_track_model import SingleTrackDiffModel
from . import util
from .util import minute


from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects



class LoiteringModelV15(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 81 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 80
    time_point_delta = 9
    window = time_points * delta

    base_filter_count = 32

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 7))
        y = input_layer
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y0 = y = Dropout(0.05)(y)
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = MaxPooling1D(4, strides=3)(y)
        y = Dropout(0.1)(y)

        depth = 2 * depth
        y = Conv1D(depth, 5)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y1 = y = Dropout(0.15)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = MaxPooling1D(4, strides=3)(y)
        y = Dropout(0.2)(y)

        depth = 2 * depth
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.25)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.3)(y)

        # Above is 1->6->19->(21)->25->76->(79)->80 (+1 for delta)
        # Below is 4 * k - 3, where k is center size
        # Above is 1->5->(21)->25->101->[(103)]-> 105



        depth = depth // 2
        y = keras.layers.UpSampling1D(size=3)(y)
        y = Dropout(0.25)(y)
        y = Concatenate()([y, 
                            keras.layers.Cropping1D((9,9))(y1)])
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.2)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)

        depth = depth // 2
        y = keras.layers.UpSampling1D(size=3)(y)
        y = Dropout(0.15)(y)
        y = keras.layers.Concatenate()([y, 
                            Cropping1D((38,38))(y0)])
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.1)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.05)(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.1)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  



class LoiteringModelV16(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 49 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 48
    time_point_delta = 4
    window = time_points * delta

    base_filter_count = 32

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    feature_padding_hours = 12.0

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 1)(y)
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y0 = y = Dropout(0)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = MaxPooling1D(3, strides=2)(y)
        y = Dropout(0)(y)

        depth *= 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y1 = y = Dropout(0.05)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = MaxPooling1D(3, strides=2)(y)
        y = Dropout(0.1)(y)

        # depth *= 2
        # y = Conv1D(depth, 3)(y)
        # y = ReLU()(y)
        # y = BatchNormalization(scale=False, momentum=0.995)(y)
        # y2 = y = Dropout(0.15)(y)
        # y = Conv1D(depth, 3)(y)
        # y = ReLU()(y)
        # y = BatchNormalization(scale=False, momentum=0.995)(y)
        # y = MaxPooling1D(3, strides=2)(y)
        # y = Dropout(0.2)(y)

        depth *= 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = Dropout(0.2)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = Dropout(0.2)(y)

        # Above is 1->5->11->(13)->15->31->(33)->35->71->(73)->76 (+8 * d, where d == delta)
        # Assume at least 4 for lower stage (d = 3 => + 3 * 8) => 100
        # (2 * n  - 3) -> (4 * n - 9) -> (8 * n - 21) at n = 4 -> 11 
        # At concats this is 4 larger, and more conveniently expressed in terms of d
        # (2 * d  + 3) -> (4 * d - 1) -> (8 * d - 9) 

        y = Reshape((-1, 1, depth))(y)
        depth //= 2
        y = Conv2DTranspose(depth, (3, 1), strides=(2, 1), padding='valid')(y)
        y = Reshape((-1, depth))(y)
        y = ReLU()(y)
        y = BatchNormalization(momentum=0.995)(y)
        y = Concatenate()([y, Cropping1D((5, 5))(y1)]) 
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = Dropout(0.125)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = Dropout(0.1)(y)

        y = Reshape((-1, 1, depth))(y)
        depth //= 2
        y = Conv2DTranspose(depth, (3, 1), strides=(2, 1), padding='valid')(y)
        y = Reshape((-1, depth))(y)
        y = ReLU()(y)
        y = BatchNormalization(momentum=0.995)(y)
        y = Concatenate()([y, Cropping1D((17, 17))(y0)]) 
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = Dropout(0.025)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, momentum=0.995)(y)
        y = Dropout(0)(y)

        # y = Reshape((-1, 1, depth))(y)
        # depth //= 2
        # y = Conv2DTranspose(depth, (3, 1), strides=(2, 1), padding='valid')(y)
        # y = Reshape((-1, depth))(y)
        # y = ReLU()(y)
        # y = BatchNormalization(momentum=0.995)(y)
        # y = Concatenate()([y, Cropping1D((41, 41))(y0)]) 
        # y = Conv1D(depth, 3)(y)
        # y = ReLU()(y)
        # y = BatchNormalization(scale=False, momentum=0.995)(y)
        # y = Dropout(0)(y)
        # y = Conv1D(depth, 3)(y)
        # y = ReLU()(y)
        # y = BatchNormalization(scale=False, momentum=0.995)(y)
        # y = Dropout(0)(y)

        y = keras.layers.Cropping1D((3, 3))(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.5)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  