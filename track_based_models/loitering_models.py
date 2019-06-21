from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, ELU, ReLU, Input, Conv1D
from keras.layers import BatchNormalization, MaxPooling1D
from keras.layers.core import Activation
from keras import optimizers
from .util import hour, minute
from .base_model import hybrid_pool_layer_2, Normalizer
from .single_track_model import SingleTrackModel
from . import util
from .util import minute, lin_interp, cos_deg, sin_deg 


class LoiteringModelV1(SingleTrackModel):
    
    delta = 10 * minute
    window = 12 * hour + delta
    time_points = window // delta

    base_filter_count = 64
    fc_nodes = 512

    data_source_lbl='transshipping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(self.time_points, 6))
        y = input_layer
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        y = Flatten()(y)
        
        y = Dense(self.fc_nodes)(y)
        y = ELU()(y)

        y = Dense(1)(y)
        y = Activation('sigmoid')(y)
        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        #opt = optimizers.Adam(lr=0.01, decay=0.5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  





class LoiteringModelV2(SingleTrackModel):
    
    delta = 10 * minute
    window = 12 * hour + delta
    time_points = window // delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 7)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 1)(y)
        y = ELU()(y)
        y = Conv1D(depth, 1)(y)
        y = ELU()(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        # 1 -> (3 -> 11) -> (23 -> 31) -> (63, 77) -> 73
        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), 1, 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
  


class LoiteringModelV3(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 93 # 72 = 12 hours, 120 = 20 hours, should be odd
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(self.time_points, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o2 = y
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o3 = y
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (4 -> 8) -> (16 -> 20) -> (40 -> 44) -> (88, 93) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 7.5 / 8
        y = keras.layers.Concatenate()([y, 
                            keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ELU()(y) # 3.5 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = keras.layers.UpSampling1D(2)(y) # 7 / 8
        y = keras.layers.Concatenate()([y, 
                            keras.layers.Cropping1D((16,16))(o2)])
        y = Conv1D(d2, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ELU()(y) # 3 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = keras.layers.UpSampling1D(2)(y) # 6 / 8
        y = keras.layers.Concatenate()([y, 
                            keras.layers.Cropping1D((40,40))(o1)])
        y = Conv1D(d1, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ELU()(y) # 2 / 4


        y = Conv1D(1, 4)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), 1, 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
  

import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x    


class LoiteringModelV4(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 93 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o2 = y
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o3 = y
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (4 -> 8) -> (16 -> 20) -> (40 -> 44) -> (88, 93) 

        # Upward branch 

        y = Conv1DTranspose(y, d3, 2, strides=2, padding='same') # 7.5 / 8
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y) # 3.5 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = Conv1DTranspose(y, d2, 2, strides=2, padding='same') # 7 / 8
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((16,16))(o2)])
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y) # 3 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = Conv1DTranspose(y, d1, 2, strides=2, padding='same') # 6 / 8
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((40,40))(o1)])
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y) # 2 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = Conv1D(1, 4)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        print(np.shape(x1), np.shape(labels))
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self



class LoiteringModelV5(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 101 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o2 = y
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o3 = y
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (5 -> 9) -> (18 -> 22) -> (44 -> 48) -> (96, 101) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y) # 3.5 / 4
        y = BatchNormalization(scale=False, center=False)(y) # 5

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((17,17))(o2)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y) # 3 / 4
        y = BatchNormalization(scale=False, center=False)(y) # 5

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((43,43))(o1)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y) # 2 / 4
        y = BatchNormalization(scale=False, center=False)(y) # 5

        y = Conv1D(1, 5)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self


class LoiteringModelV6(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 101 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        input_layer = Input(shape=(None, 6))

        d1 = depth = self.base_filter_count        
        y = input_layer
        y = Conv1D(depth, 4, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = o2 = Dropout(0.1)(y)
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = o3 = Dropout(0.2)(y)
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Dropout(0.3)(y)


        # (5 -> 9) -> (18 -> 22) -> (44 -> 48) -> (96, 101) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((17,17))(o2)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((43,43))(o1)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)

        y = Conv1D(1, 5)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05,
                               clipnorm=1.)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
    


class LoiteringModelV7(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 101 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        input_layer = Input(shape=(None, 6))

        d1 = depth = self.base_filter_count        
        y = input_layer
        y = Conv1D(depth, 4, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o1 = Dropout(0.1)(y)
        y = MaxPooling1D(2)(y)
        
        d2 = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o2 = Dropout(0.2)(y)
        y = MaxPooling1D(2)(y)
        
        d3 = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o3 = Dropout(0.3)(y)
        y = MaxPooling1D(2)(y)

        depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Dropout(0.4)(y)


        # (5 -> 9) -> (18 -> 22) -> (44 -> 48) -> (96, 101) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((17,17))(o2)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((43,43))(o1)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)

        y = Conv1D(1, 5)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05,
                               clipnorm=0.5)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
    