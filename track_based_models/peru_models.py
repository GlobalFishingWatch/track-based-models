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

from .single_track_model import SingleTrackDiffModel
from . import util
from .util import minute, lin_interp, cos_deg, sin_deg 


from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects




class Model(SingleTrackDiffModel):

    delta = 10 * minute
    time_points = 81 # 72 = 12 hours, 120 = 20 hours, should be odd
    internal_time_points = 80
    time_point_delta = 4
    window = time_points * delta

    base_filter_count = 32

    data_source_lbl='fishing' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute

    vessel_label = 'position_data'

    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 7))
        y = input_layer
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y0 = y = Dropout(0.1)(y)
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = MaxPooling1D(4, strides=3)(y)
        y = Dropout(0.3)(y)

        depth *= 2
        y = Conv1D(depth, 5)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y1 = y = Dropout(0.3)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = MaxPooling1D(4, strides=3)(y)
        y = Dropout(0.5)(y)

        depth *= 2
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.5)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.5)(y)

        # Above is 1->6->19->(21)->25->76->(79)->80 (+1 for delta)
        # Below is 4 * k - 3, where k is center size
        # Above is 1->5->(21)->25->101->[(103)]-> 105



        depth //= 2
        y = keras.layers.UpSampling1D(size=3)(y)
        y = Dropout(0.3)(y)
        y = Concatenate()([y, 
                            keras.layers.Cropping1D((9,9))(y1)])
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.3)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)

        depth //= 2
        y = keras.layers.UpSampling1D(size=3)(y)
        y = Dropout(0.1)(y)
        y = keras.layers.Concatenate()([y, 
                            Cropping1D((38,38))(y0)]) 
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.1)(y)
        y = Conv1D(depth, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False, momentum=0.995)(y)
        y = Dropout(0.1)(y)
        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.1)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))

        depth = np.clip(f.depth, 0, 200)
        logged_depth = np.log(1 + depth) + 40 * noise

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)), 
                             np.sin(np.radians(f.angle_feature)),
                             f.dir_a,
                             f.dir_b,
                             np.exp(-f.delta_time),
                             logged_depth, 
                             ]), angle


    # @classmethod
    # def cook_features(cls, raw_features, angle=None, noise=None):
    #     speed = raw_features[:, 0]
    #     angle = np.random.uniform(0, 360) if (angle is None) else angle
    #     radians = np.radians(angle)
    #     angle_feat = angle + (90 - raw_features[:, 1])
        
    #     ndx = np.random.randint(len(raw_features))
    #     lat0 = raw_features[ndx, 2]
    #     lon0 = raw_features[ndx, 3]
    #     lat = raw_features[:, 2] 
    #     lon = raw_features[:, 3] 
    #     scale = np.cos(np.radians(lat))
    #     d1 = lat - lat0
    #     d2 = (lon - lon0) * scale
    #     dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
    #     dir_b = np.cos(radians) * d1 + np.sin(radians) * d2
    #     depth = -raw_features[:, 5]
    #     distance = raw_features[:, 6]

    #     noise1 = noise2 = noise
    #     if noise is None:
    #         noise1 = np.random.normal(0, .05, size=len(raw_features[:, 4]))
    #         noise2 = np.random.normal(0, .05, size=len(raw_features[:, 4]))

    #     noisy_time = np.maximum(raw_features[:, 4] / 
    #                             float(cls.data_far_time) + noise1, 0)

    #     depth = np.clip(depth, 0, 200)
    #     logged_depth = np.log(1 + depth) + 40 * noise2

    #     is_far = np.exp(-noisy_time) 
    #     return np.transpose([speed,
    #                          np.cos(np.radians(angle_feat)), 
    #                          np.sin(np.radians(angle_feat)),
    #                          dir_a,
    #                          dir_b,
    #                          is_far,
    #                          logged_depth, 
    #                          ]), angle

    # @classmethod
    # def cook_features(cls, raw_features, angle=None, noise=None):
    #     speed = raw_features[:, 0]
    #     angle = np.random.uniform(0, 2*np.pi) if (angle is None) else angle
    #     angle_feat = angle + (np.pi / 2.0 - raw_features[:, 1])
        
    #     ndx = np.random.randint(len(raw_features))
    #     lat0 = raw_features[ndx, 2]
    #     lon0 = raw_features[ndx, 3]
    #     lat = raw_features[:, 2] 
    #     lon = raw_features[:, 3] 
    #     scale = np.cos(np.radians(lat))
    #     d1 = lat - lat0
    #     d2 = (lon - lon0) * scale
    #     dir_a = np.cos(angle) * d2 - np.sin(angle) * d1
    #     dir_b = np.cos(angle) * d1 + np.sin(angle) * d2
    #     depth = -raw_features[:, 5]
    #     distance = raw_features[:, 6]


    #     noise1 = noise2 = noise
    #     if noise is None:
    #         noise1 = np.random.normal(0, .05, size=len(raw_features[:, 4]))
    #         noise2 = np.random.normal(0, .05, size=len(raw_features[:, 4]))

    #     noisy_time = np.maximum(raw_features[:, 4] / 
    #                             float(cls.data_far_time) + noise1, 0)

    #     depth = np.clip(depth, 0, 200)
    #     logged_depth = np.log(1 + depth) + 40 * noise2

    #     is_far = np.exp(-noisy_time) 
    #     return np.transpose([speed,
    #                          np.cos(angle_feat), 
    #                          np.sin(angle_feat),
    #                          dir_a,
    #                          dir_b,
    #                          0 * is_far,
    #                          logged_depth, 
    #                          ]), angle
