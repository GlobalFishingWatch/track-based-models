from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, ELU, ReLU, Input, Conv1D, maximum
from keras.layers import BatchNormalization, MaxPooling1D, Concatenate, Maximum, DepthwiseConv2D
from keras.layers import Cropping1D, AveragePooling1D, Cropping1D, Conv2DTranspose, SeparableConv1D
from keras.layers import Conv3D, ZeroPadding3D, MaxPooling3D, Lambda
from keras.layers.core import Activation, Reshape
from keras import optimizers
import logging
from .util import hour, minute
from .base_model import hybrid_pool_layer_2, Normalizer
from .single_track_model import SingleTrackDiffModel, SingleTrackDistModel
from . import util
from .util import minute


from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects


class LoiteringModelV3D1(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 159 # 144 = 12 hours,
    internal_time_points = 158
    time_point_delta = 1
    window = time_points * delta

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    max_speed = 20
    delta_speed = 2
    n2d = 31
    max_speed = 20.0
    delta_speed = 2 * max_speed / (n2d - 1)

    def __init__(self, width=None):
        
        self.normalizer = None
        
        pool_width = 3
        dilation = 1

        def batch_norm(y, center=True, scale=False):
            return BatchNormalization(momentum=0.99, center=center, scale=scale)(y)

        def conv3d(y, n_filters, kernel=3, strides=(1, 1, 1), pad=True):
            if pad:
                y = ZeroPadding3D((0, 1, 1))(y)
            y = Conv3D(n_filters, kernel, strides=strides, activation='relu')(y)
            return batch_norm(y)

        def block_3d(y, nf0):
            y = conv3d(y, nf0)
            y = conv3d(y, nf0)
            y = conv3d(y, 2 * nf0, strides=(1, 2, 2), pad=False)
            return y

        def block_a(y, n):
            dr = 2**n
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            return MaxPooling1D(2 * dr + 1, strides=1)(y), y

        def block_b(y, n):
            dr = 2**n
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            return batch_norm(y, scale=False)

        def crop(y, n):
            return Cropping1D((n, n))(y)

        input_layer = Input(shape=(width, self.n2d, self.n2d, 2))
        y = conv3d(input_layer, 8, (2, 3, 3)) 
        y = batch_norm(y, scale=False)

        y = block_3d(y, 8) # -> 16 x 15
        y = block_3d(y, 16) # -> 32 x 7
        y = block_3d(y, 32) # -> 64 x 3
        y = block_3d(y, 64) # -> 128 x 1, 8
 
        w = -1 if (width is None) else (width - 25)
        y = Reshape((w, 128))(y)

        depth = 128

        y, y0 = block_a(y, 0) # 6 
        y, y1 = block_a(y, 1) # 6 + 12
        y, y2 = block_a(y, 2) # 6 + 12 + 24
        y, y3 = block_a(y, 3) # 6 + 12 + 24 + 48 = 90

        y = Concatenate()([y, crop(y2, 28)])
        y = block_b(y, 2) # 90 + 24
        y = Concatenate()([y, crop(y1, 50)])
        y = block_b(y, 1) # 90 + 24 + 12
        y = Concatenate()([y, crop(y0, 61)])
        y = block_b(y, 0) # 90 + 24 + 12 + 6

        y = Dropout(0.5)(y) # 132 + 25 from top = 157

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        self.optimizer = opt

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        cooked = np.zeros([len(f.speed), cls.n2d, cls.n2d, 2], np.float32)

        hours = cls.delta / hour

        def proc(v):
            i = np.round(v / cls.delta_speed).astype(int) + cls.n2d // 2
            return np.clip(i, 0, cls.n2d - 1)

        x = proc(f.dir_a * 60 / hours)
        y = proc(f.dir_b * 60 / hours)
        sx = proc(f.speed * np.cos(np.radians(f.angle_feature)))
        sy = proc(f.speed * np.sin(np.radians(f.angle_feature))) 
        inds = np.arange(len(cooked))

        cooked[inds, x, y, np.zeros(len(cooked), dtype=int)] = 1
        cooked[inds, sx, sy, np.ones(len(cooked), dtype=int)] = 1

        return cooked, angle

    def preprocess(self, x, fit=False):
        x = np.asarray(x[:, 1:]) 
        return x


class LoiteringModelVD1(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 109 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 108
    time_point_delta = 1
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

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y0 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        pool_width *= 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y1 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        pool_width *= 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y2 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)



        depth = depth * 2 // 3
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((22,21))(y2)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((40,40))(y1)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((49,49))(y0)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

class LoiteringModelVD2(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 139
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 128

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y0 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        pool_width *= 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y1 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        pool_width *= 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y2 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        dilation *= 2
        y = Conv1D(depth, 5, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 5, dilation_rate=dilation)(y)
        y = ReLU()(y)



        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((38,37))(y2)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((56,56))(y1)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((65,65))(y0)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)
        self.optimizer = opt
        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def preprocess(self, x, fit=False):
        if fit:
            # Build a normalizer for compatibility
            self.normalizer = Normalizer().fit(x)
        # Skip fitting.
        return np.asarray(x)[:, 2:] 

class LoiteringModelVD2c(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 145 # 144 = 12 hours,
    internal_time_points = 143
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 128

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y0 = y = BatchNormalization()(y)
        y = Cropping1D([pool_width//2, pool_width//2])(y0)

        depth = depth * 3 // 2
        pool_width  = pool_width * 2 + 1
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y1 = y = BatchNormalization()(y)
        y = Cropping1D([pool_width//2, pool_width//2])(y)
        depth = depth * 3 // 2
        pool_width  = pool_width * 2 + 1
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y2 = y = BatchNormalization()(y)
        y = Cropping1D([pool_width//2, pool_width//2])(y)
        depth = depth * 3 // 2
        dilation *= 2
        y = Conv1D(depth, 5, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 5, dilation_rate=dilation)(y)
        y = ReLU()(y)



        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((39,39))(y2)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((58,58))(y1)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((67,67))(y0)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)
        self.optimizer = opt
        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def preprocess(self, x, fit=False):
        if fit:
            # Build a normalizer for compatibility
            self.normalizer = Normalizer().fit(x)
        # Skip fitting.
        return np.asarray(x)[:, 2:] 


class LoiteringModelVD2b(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 139
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 128

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y0 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        pool_width = 2 * pool_width - 1
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y1 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        pool_width = 2 * pool_width - 1
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y2 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 3 // 2
        dilation *= 2
        y = Conv1D(depth, 5, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 5, dilation_rate=dilation)(y)
        y = ReLU()(y)



        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((38,37))(y2)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((56,56))(y1)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth * 2 // 3 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((65,65))(y0)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)
        self.optimizer = opt
        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  


class LoiteringModelVD3(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 140
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y0 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 2
        pool_width *= 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y1 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 2
        pool_width *= 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y2 = y = BatchNormalization()(y)
        y = MaxPooling1D(pool_width, strides=1)(y)

        depth = depth * 2
        dilation *= 2
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth // 2 
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((38,37))(y2)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth // 2
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((56,56))(y1)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth = depth // 2
        dilation //= 2
        y = Concatenate()([y, keras.layers.Cropping1D((65,65))(y0)])
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=dilation)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  


class LoiteringModelVD4(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 151 # 144 = 12 hours,
    internal_time_points = 150
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        depth = self.base_filter_count

        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 2, dilation_rate=1, activation='relu')(y)
        y = BatchNormalization()(y)

        y = Conv1D(depth, 3, dilation_rate=1, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=2, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=4, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=8, activation='relu')(y)
        y = BatchNormalization()(y)

        y = Conv1D(depth, 3, dilation_rate=1, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=2, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=4, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=8, activation='relu')(y)
        y = BatchNormalization()(y)

        y = Conv1D(depth, 3, dilation_rate=1, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=2, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=4, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=8, activation='relu')(y)
        y = BatchNormalization()(y)

        y = Conv1D(depth, 3, dilation_rate=1, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=2, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=4, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=8, activation='relu')(y)
        y = BatchNormalization()(y)

        y = Conv1D(depth, 3, dilation_rate=4, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=4, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=2, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=2, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=1, activation='relu')(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3, dilation_rate=1, activation='relu')(y)
        y = BatchNormalization()(y)

        y = Dropout(0.5)(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        self.optimizer = opt
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

class LoiteringModelVD5(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 151 # 144 = 12 hours,
    internal_time_points = 150
    time_point_delta = 1
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

    def __init__(self, width=None):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        d2 = 2 * depth

        def block(x, n1, n2, dr):
            dx = SeparableConv1D(n2, 3, dilation_rate=dr)(x)
            dx = BatchNormalization()(dx)
            dx = ReLU()(dx)
            dx = Conv1D(n1, 1, dilation_rate=1)(dx)
            x = keras.layers.Add()([keras.layers.Cropping1D((dr, dr))(x),  dx])
            x = ReLU()(x)
            x = BatchNormalization()(x)
            return x


        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 2, dilation_rate=1)(y)
        y = BatchNormalization()(y)
        y = ReLU()(y)

        y = block(y, depth, d2, 1)        
        y = block(y, depth, d2, 2)
        y = block(y, depth, d2, 4)
        y = block(y, depth, d2, 8)

        y = block(y, depth, d2, 1)        
        y = block(y, depth, d2, 2)
        y = block(y, depth, d2, 4)
        y = block(y, depth, d2, 8)

        y = block(y, depth, d2, 1)        
        y = block(y, depth, d2, 2)
        y = block(y, depth, d2, 4)
        y = block(y, depth, d2, 8)

        y = block(y, depth, d2, 1)        
        y = block(y, depth, d2, 2)
        y = block(y, depth, d2, 4)
        y = block(y, depth, d2, 8)

        y = block(y, depth, d2, 4)
        y = block(y, depth, d2, 2)
        y = block(y, depth, d2, 1)        

        y = block(y, depth, d2, 4)
        y = block(y, depth, d2, 2)
        y = block(y, depth, d2, 1)        

        y = Dropout(0.5)(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  


class LoiteringModelVD6(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 151 # 144 = 12 hours,
    internal_time_points = 150
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        nmax = 3

        def maxout(n, dr, x):
            ys = [SeparableConv1D(n, 3, dilation_rate=dr)(x) for _ in range(nmax)]
            return Maximum()(ys)

        def block(x, n, dr):
            dx = maxout(n, dr, x)
            dx = BatchNormalization()(dx)
            dx = Conv1D(n, 3, dilation_rate=dr)(dx)
            x = keras.layers.Add()([keras.layers.Cropping1D((2 * dr, 2 * dr))(x),  dx])
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x


        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 2, dilation_rate=1)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        y = block(y, depth, 1)        
        y = block(y, depth, 2)
        y = block(y, depth, 4)
        y = block(y, depth, 8)

        y = block(y, depth, 8)
        y = block(y, depth, 4)
        y = block(y, depth, 2)
        y = block(y, depth, 1)        

        y = block(y, depth, 4)
        y = block(y, depth, 2)
        y = block(y, depth, 1)               

        y = Dropout(0.5)(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  


class LoiteringModelVD7(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 140
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        def batch_norm(y, center=True, scale=True):
            return BatchNormalization(momentum=0.99, center=center, scale=scale)(y)

        input_layer = Input(shape=(width, 7))
        y = Conv1D(depth, 2, activation='relu')(input_layer)
        y = batch_norm(y, scale=False)


        def block_a(y, n):
            dr = 2**n
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            return MaxPooling1D(2 * dr + 1, strides=1)(y), y

        def block_b(y, n):
            dr = 2**n
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = Conv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            return batch_norm(y, scale=False)

        # Output size drops by 6 * dr = 6 * 2 ** n

        y, y0 = block_a(y, 0) # 6 
        y, y1 = block_a(y, 1) # 6 + 12
        y, y2 = block_a(y, 2) # 6 + 12 + 24
        y, y3 = block_a(y, 3) # 6 + 12 + 24 + 48 = 90

        def crop(y, n):
            return Cropping1D((n, n))(y)

        y = Concatenate()([y, crop(y2, 28)])
        y = block_b(y, 2) # 90 + 24
        y = Concatenate()([y, crop(y1, 50)])
        y = block_b(y, 1) # 90 + 24 + 12
        y = Concatenate()([y, crop(y0, 61)])
        y = block_b(y, 0) # 90 + 24 + 12 + 6

        y = block_b(y, 0) # 90 + 24 + 12 + 6 + 6

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=1.0, momentum=0.9, 
        #                                 decay=0.99, nesterov=True)
        self.optimizer = opt

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             f.speed * np.cos(np.radians(f.angle_feature)) * dt, 
                             f.speed * np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time),
                             0 * logged_depth, 
                             ]), angle


class LoiteringModelVD9(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 140
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        def batch_norm(y, center=True, scale=True):
            return BatchNormalization(momentum=0.99, center=center, scale=scale)(y)

        def block_a(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            return MaxPooling1D(2 * dr + 1, strides=1)(y), y

        def block_b(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            return batch_norm(y, scale=False)

        input_layer = Input(shape=(width, 4))
        y = Conv1D(depth, 2, activation='relu')(input_layer)
        y = batch_norm(y, scale=False)

        # Output size drops by 6 * dr = 6 * 2 ** n

        y, y0 = block_a(y, 0) # 6 
        y, y1 = block_a(y, 1) # 6 + 12
        y, y2 = block_a(y, 2) # 6 + 12 + 24
        y, y3 = block_a(y, 3) # 6 + 12 + 24 + 48 = 90

        def crop(y, n):
            return Cropping1D((n, n))(y)

        y = Concatenate()([y, crop(y2, 28)])
        y = block_b(y, 2) # 90 + 24
        y = Concatenate()([y, crop(y1, 50)])
        y = block_b(y, 1) # 90 + 24 + 12
        y = Concatenate()([y, crop(y0, 61)])
        y = block_b(y, 0) # 90 + 24 + 12 + 6

        y = block_b(y, 0) # 90 + 24 + 12 + 6 + 6

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        self.optimizer = opt

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        imp_speed = np.hypot(f.dir_a * 60 / dt, f.dir_b * 60 / dt)
        # print('imp', imp_speed.mean(), imp_speed.std(), imp_speed.max())
        # print('rep', f.speed.mean(), f.speed.std(), f.speed.max())
        imp_speed = np.clip(imp_speed, 0, 30) / 30
        speed = np.clip(f.speed, 0, 30) / 30

        return np.transpose([speed , # Reported speed
                             imp_speed, # Implied speed
                             np.cos(np.radians(f.delta_degrees)), # Reported course 
                             np.sin(np.radians(f.delta_degrees)), # Reported course                    
                             ]), angle
        
    def preprocess(self, x, fit=False):
        x = np.asarray(x)[:, 1:]
        return x

class LoiteringModelVD10(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 140
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        def batch_norm(y, center=True, scale=True):
            return BatchNormalization(momentum=0.99, center=center, scale=scale)(y)

        def block_a(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            return MaxPooling1D(2 * dr + 1, strides=1)(y), y

        def block_b(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            return batch_norm(y, scale=False)

        input_layer = Input(shape=(width, 3))
        y = Conv1D(depth, 2, activation='relu')(input_layer)
        y = batch_norm(y, scale=False)

        # Output size drops by 6 * dr = 6 * 2 ** n

        y, y0 = block_a(y, 0) # 6 
        y, y1 = block_a(y, 1) # 6 + 12
        y, y2 = block_a(y, 2) # 6 + 12 + 24
        y, y3 = block_a(y, 3) # 6 + 12 + 24 + 48 = 90

        def crop(y, n):
            return Cropping1D((n, n))(y)

        y = Concatenate()([y, crop(y2, 28)])
        y = block_b(y, 2) # 90 + 24
        y = Concatenate()([y, crop(y1, 50)])
        y = block_b(y, 1) # 90 + 24 + 12
        y = Concatenate()([y, crop(y0, 61)])
        y = block_b(y, 0) # 90 + 24 + 12 + 6

        y = block_b(y, 0) # 90 + 24 + 12 + 6 + 6

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        self.optimizer = opt

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        imp_speed = np.hypot(f.dir_a * 60 / dt, f.dir_b * 60 / dt)
        imp_speed = np.clip(imp_speed, 0, 30) / 30

        return np.transpose([imp_speed, # Implied speed
                             np.cos(np.radians(f.delta_degrees)), # Reported course 
                             np.sin(np.radians(f.delta_degrees)), # Reported course                    
                             ]), angle
        
    def preprocess(self, x, fit=False):
        x = np.asarray(x)[:, 1:]
        return x

class LoiteringModelVD11(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 140
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        def batch_norm(y, center=True, scale=True):
            return BatchNormalization(momentum=0.99, center=center, scale=scale)(y)

        def block_a(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            return MaxPooling1D(2 * dr + 1, strides=1)(y), y

        def block_b(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            return batch_norm(y, scale=False)

        input_layer = Input(shape=(width, 3))
        y = Conv1D(depth, 2, activation='relu')(input_layer)
        y = batch_norm(y, scale=False)

        # Output size drops by 6 * dr = 6 * 2 ** n

        y, y0 = block_a(y, 0) # 6 
        y, y1 = block_a(y, 1) # 6 + 12
        y, y2 = block_a(y, 2) # 6 + 12 + 24
        y, y3 = block_a(y, 3) # 6 + 12 + 24 + 48 = 90

        def crop(y, n):
            return Cropping1D((n, n))(y)

        y = Concatenate()([y, crop(y2, 28)])
        y = block_b(y, 2) # 90 + 24
        y = Concatenate()([y, crop(y1, 50)])
        y = block_b(y, 1) # 90 + 24 + 12
        y = Concatenate()([y, crop(y0, 61)])
        y = block_b(y, 0) # 90 + 24 + 12 + 6

        y = block_b(y, 0) # 90 + 24 + 12 + 6 + 6

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        self.optimizer = opt

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        speed = np.clip(f.speed, 0, 30) / 30

        return np.transpose([speed, # Reported speed
                             np.cos(np.radians(f.delta_degrees)), # Reported course 
                             np.sin(np.radians(f.delta_degrees)), # Reported course                    
                             ]), angle
        
    def preprocess(self, x, fit=False):
        x = np.asarray(x)[:, 1:]
        return x


class LoiteringModelVD8(SingleTrackDiffModel):
    
    delta = 5 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 140
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        def batch_norm(y, center=True, scale=True):
            return BatchNormalization(momentum=0.99, center=center, scale=scale)(y)

        def block_a(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            return MaxPooling1D(2 * dr + 1, strides=1)(y), y

        def block_b(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            return batch_norm(y, scale=False)

        input_layer = Input(shape=(width, 4))
        y = Conv1D(depth, 2, activation='relu')(input_layer)
        y = batch_norm(y, scale=False)

        # Output size drops by 6 * dr = 6 * 2 ** n

        y, y0 = block_a(y, 0) # 6 
        y, y1 = block_a(y, 1) # 6 + 12
        y, y2 = block_a(y, 2) # 6 + 12 + 24
        y, y3 = block_a(y, 3) # 6 + 12 + 24 + 48 = 90

        def crop(y, n):
            return Cropping1D((n, n))(y)

        y = Concatenate()([y, crop(y2, 28)])
        y = block_b(y, 2) # 90 + 24
        y = Concatenate()([y, crop(y1, 50)])
        y = block_b(y, 1) # 90 + 24 + 12
        y = Concatenate()([y, crop(y0, 61)])
        y = block_b(y, 0) # 90 + 24 + 12 + 6

        y = block_b(y, 0) # 90 + 24 + 12 + 6 + 6

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        self.optimizer = opt

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        imp_speed = np.hypot(f.dir_a * 60 / dt, f.dir_b * 60 / dt)

        return np.transpose([f.speed / 30, # Reported speed
                             imp_speed / 30, # Implied speed
                             np.cos(np.radians(f.delta_degrees)), # Reported course 
                             np.sin(np.radians(f.delta_degrees)), # Reported course                    
                             ]), angle
        
    def preprocess(self, x, fit=False):
        x = np.asarray(x)[:, 1:]
        return x





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
        y = BatchNormalization()(y)
        y0 = y = Dropout(0.05)(y)
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = MaxPooling1D(4, strides=3)(y)
        y = Dropout(0.1)(y)

        depth *= 2
        y = Conv1D(depth, 5)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y1 = y = Dropout(0.15)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = MaxPooling1D(4, strides=3)(y)
        y = Dropout(0.2)(y)

        depth *= 2
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.25)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.3)(y)

        # Above is 1->6->19->(21)->25->76->(79)->80 (+1 for delta)
        # Below is 4 * k - 3, where k is center size
        # Above is 1->5->(21)->25->101->[(103)]-> 105

        depth //= 2
        y = keras.layers.UpSampling1D(size=3)(y)
        y = Dropout(0.25)(y)
        y = Concatenate()([y, keras.layers.Cropping1D((9,9))(y1)])
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.2)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth //= 2
        y = keras.layers.UpSampling1D(size=3)(y)
        y = Dropout(0.15)(y)
        y = keras.layers.Concatenate()([y, Cropping1D((38,38))(y0)])
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.1)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
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


class LoiteringModelV15D(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 91 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 90
    time_point_delta = 1
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

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y0 = y = Dropout(0.05)(y)
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = MaxPooling1D(4, strides=1)(y)
        y = Dropout(0.1)(y)

        depth *= 2
        y = Conv1D(depth, 5, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y1 = y = Dropout(0.15)(y)
        y = Conv1D(depth, 3, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = MaxPooling1D(12, strides=1)(y)
        y = Dropout(0.2)(y)

        depth *= 2
        y = Conv1D(depth, 4, dilation_rate=9)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.25)(y)
        y = Conv1D(depth, 3, dilation_rate=9)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.3)(y)

        depth //= 2
        y = Dropout(0.25)(y)
        y = Concatenate()([y, keras.layers.Cropping1D((31,31))(y1)])
        y = Conv1D(depth, 2, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.2)(y)
        y = Conv1D(depth, 2, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth //= 2
        y = Dropout(0.15)(y)
        y = keras.layers.Concatenate()([y, Cropping1D((43,43))(y0)])
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = Dropout(0.1)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
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



class LoiteringModelV15D2(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 91 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 90
    time_point_delta = 1
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

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y0 = y# = Dropout(0.05)(y)
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = MaxPooling1D(4, strides=1)(y)
        #y = Dropout(0.1)(y)

        depth *= 2
        y = Conv1D(depth, 5, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y1 = y #= Dropout(0.15)(y)
        y = Conv1D(depth, 3, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        y = MaxPooling1D(12, strides=1)(y)
        #y = Dropout(0.2)(y)

        depth *= 2
        y = Conv1D(depth, 4, dilation_rate=9)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        #y = Dropout(0.25)(y)
        y = Conv1D(depth, 3, dilation_rate=9)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        #y = Dropout(0.3)(y)

        depth //= 2
        #y = Dropout(0.25)(y)
        y = Concatenate()([y, keras.layers.Cropping1D((31,31))(y1)])
        y = Conv1D(depth, 2, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        #y = Dropout(0.2)(y)
        y = Conv1D(depth, 2, dilation_rate=3)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)

        depth //= 2
        #y = Dropout(0.15)(y)
        y = keras.layers.Concatenate()([y, Cropping1D((43,43))(y0)])
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        #y = Dropout(0.1)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization()(y)
        #y = Dropout(0.05)(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  



class LoiteringModelV16(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 93 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 92
    time_point_delta = 8
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

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 1)(y)
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y0 = y = Dropout(0)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = MaxPooling1D(3, strides=2)(y)
        y = Dropout(0)(y)

        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y1 = y = Dropout(0.05)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = MaxPooling1D(3, strides=2)(y)
        y = Dropout(0.1)(y)

        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y2 = y = Dropout(0.15)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = MaxPooling1D(3, strides=2)(y)
        y = Dropout(0.2)(y)

        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.2)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.2)(y)

        # Above is 1->5->11->(13)->15->31->(33)->35->71->(73)->76 (+8 * d, where d == delta)
        # Assume at least 4 for lower stage (d = 2 => + 2 * 8) => 92
        # (2 * n  - 3) -> (4 * n - 9) -> (8 * n - 21) at n = 3 -> 3 
        # At concats this is 4 larger, and more conveniently expressed in terms of d
        # (2 * d  + 3) -> (4 * d - 1) -> (8 * d - 9) 

        y = Reshape((-1, 1, depth))(y)
        depth = 2 * depth // 3
        y = Conv2DTranspose(depth, (3, 1), strides=(2, 1), padding='valid')(y)
        y = Reshape((-1, depth))(y)
        y = ReLU()(y)
        y = BatchNormalization(momentum=0.995)(y)
        y = Concatenate()([y, Cropping1D((5, 5))(y2)]) 
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.125)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)

        y = Reshape((-1, 1, depth))(y)
        depth = 2 * depth // 3
        y = Conv2DTranspose(depth, (3, 1), strides=(2, 1), padding='valid')(y)
        y = Reshape((-1, depth))(y)
        y = ReLU()(y)
        y = BatchNormalization(momentum=0.995)(y)
        y = Concatenate()([y, Cropping1D((17, 17))(y1)]) 
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.025)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0)(y)

        y = Reshape((-1, 1, depth))(y)
        depth = 2 * depth // 3
        y = Conv2DTranspose(depth, (3, 1), strides=(2, 1), padding='valid')(y)
        y = Reshape((-1, depth))(y)
        y = ReLU()(y)
        y = BatchNormalization(momentum=0.995)(y)
        y = Concatenate()([y, Cropping1D((41, 41))(y0)]) 
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0)(y)

        y = keras.layers.Cropping1D((1, 1))(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.5)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  




class LoiteringModelV18(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 91 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 90
    time_point_delta = 1
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
        y = Conv1D(depth, 2, activation='relu')(y)

        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.05)(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = y0 = Dropout(0.05)(y)

        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = y1 = Dropout(0.1)(y)

        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.15)(y)
        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = y2 = Dropout(0.15)(y)

        y = Conv1D(8 * depth, 3, activation='relu', dilation_rate=8)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.2)(y)
        y = Conv1D(8 * depth, 3, activation='relu', dilation_rate=8)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.2)(y)

        y = Concatenate()([y, Cropping1D((16, 16))(y2)]) 
        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.15)(y)
        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.15)(y)

        y = Concatenate()([y, Cropping1D((32, 32))(y1)]) 
        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)

        y = Concatenate()([y, Cropping1D((40, 40))(y0)]) 
        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.05)(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.05)(y)

        # 1 -> 5 -> 13 -> 29 -> 61 -> 77 -> 85 -> 89 -> 90

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.9)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model 


class LoiteringModelV19(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 91 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 90
    time_point_delta = 1
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
        y = Conv1D(depth, 2, activation='relu')(y)

        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.05)(y)
        ym = MaxPooling1D(3, strides=1)(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = Concatenate()([y, ym])
        y = y0 = Dropout(0.05)(y)

        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        ym = MaxPooling1D(5, strides=1)(y)
        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = Concatenate()([y, ym])
        y = y1 = Dropout(0.1)(y)

        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.15)(y)
        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = y2 = Dropout(0.15)(y)

        y = Conv1D(8 * depth, 3, activation='relu', dilation_rate=8)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.2)(y)
        y = Conv1D(8 * depth, 3, activation='relu', dilation_rate=8)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.2)(y)

        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.15)(y)
        y = Conv1D(4 * depth, 3, activation='relu', dilation_rate=4)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.15)(y)

        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = Conv1D(2 * depth, 3, activation='relu', dilation_rate=2)(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)

        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.05)(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.05)(y)

        # 1 -> 5 -> 13 -> 29 -> 61 -> 77 -> 85 -> 89 -> 90

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.9)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model 



class LoiteringModelV20(SingleTrackDiffModel):
    
    delta = 10 * minute
    time_points = 85 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 84
    time_point_delta = 1
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
        
        def maxout(depth, dilation, y):
            y0 = Conv1D(depth, 3, dilation_rate=dilation)(y)
            y1 = Conv1D(depth, 3, dilation_rate=dilation)(y)
            return Maximum()([y0, y1])

        input_layer = Input(shape=(width, 7))
        y = input_layer
        y = Conv1D(depth, 2, activation='relu')(y)

        y = maxout(depth, 1, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(depth, 2, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(depth, 4, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(depth, 8, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y) # 30

        y = maxout(2 * depth, 1, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(2 * depth, 2, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(2 * depth, 4, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(2 * depth, 8, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y) # 30

        y = maxout(3 * depth, 1, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(3 * depth, 2, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(3 * depth, 4, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y) # 14

        y = maxout(2 * depth, 1, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y)
        y = maxout(2 * depth, 2, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y) # 6

        y = maxout(depth, 2, y)
        y = BatchNormalization(scale=False)(y)
        y = Dropout(0.1)(y) # 2

        # 2 + 6 + 14 + 30 + 30 + 1 = 83

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.9)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model 



class LoiteringModelDistV1(SingleTrackDistModel):
    
    delta = 5 * minute
    time_points = 139 # 144 = 12 hours,
    internal_time_points = time_points
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    vessel_label = 'position_data_reefer'

    def __init__(self, width=None):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        def batch_norm(y, center=True, scale=True):
            return BatchNormalization(momentum=0.99, center=center, scale=scale)(y)

        def distance_block(y, dr):
            n = 2 * dr
            def distance(x):
                dlon = x[:, n:, 2:3] - x[:, :-n, 2:3]
                dlat = x[:, n:, 3:4] - x[:, :-n, 3:4]
                avglat = 0.5 * (x[:, n:, 3:4] - x[:, :-n, 3:4])
                scale = K.cos(avglat)
                dist =  ((scale * dlon) ** 2 + dlat ** 2) ** 0.5
                dcourse = x[:, n:, 1:2] - x[:, :-n, 1:2]
                return K.concatenate([K.cos(dcourse), K.sin(dcourse), dist])
            def output_shape(input_shape):
                shape = list(input_shape)
                if shape[-2] is not None:
                    shape[-2] -= n
                shape[-1] = 3
                return tuple(shape)            
            return Lambda(distance, output_shape=output_shape)(y)


        def block_a(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = Concatenate(axis=-1)([y, distance_block(yi, dr)])
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            return y

        def block_b(y, n):
            dr = 2**n
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            y = batch_norm(y, scale=False)
            y = SeparableConv1D(depth, 3, dilation_rate=dr, activation='relu')(y)
            return batch_norm(y, scale=False)

        def crop(y, n):
            return Cropping1D((n, n))(y)

        input_layer = Input(shape=(width, 4))
        yi = input_layer

        # Speed only
        y = Lambda(lambda x: x[:,:,:1], output_shape=lambda x: x[:2] + (1,))(input_layer)

        # Output size drops by 6 * dr = 6 * 2 ** n

        y = y0 = block_a(y, 0) # 6 
        yi = crop(yi, 3)
        y = y1 = block_a(y, 1) # 6 + 12
        yi = crop(yi, 6)
        y = y2 = block_a(y, 2) # 6 + 12 + 24
        yi = crop(yi, 12)
        y = y3 = block_a(y, 3) # 6 + 12 + 24 + 48 = 90


        y = Concatenate()([y, crop(y2, 24)])
        y = block_b(y, 2) # 90 + 24
        y = Concatenate()([y, crop(y1, 48)])
        y = block_b(y, 1) # 90 + 24 + 12
        y = Concatenate()([y, crop(y0, 60)])
        y = block_b(y, 0) # 90 + 24 + 12 + 6

        y = block_b(y, 0) # 90 + 24 + 12 + 6 + 6

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam()
        self.optimizer = opt

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

        
    def preprocess(self, x, fit=False):
        x = np.asarray(x)
        return x
 