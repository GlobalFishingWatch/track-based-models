from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, ELU, Input, Conv1D
from keras.layers import BatchNormalization
from keras.layers.core import Activation
from keras import optimizers
from .util import hour, minute
from .base_model import hybrid_pool_layer_2
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
        
        input_layer = Input(shape=(self.time_points, 6))
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
        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer[:, :, 0])
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  



  






    


