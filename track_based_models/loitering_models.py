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





    @classmethod
    def generate_data(cls, paths, min_samples, label_window=None, seed=888, 
                    skip_label=False, keep_fracs=(1,), noise=None, 
                    precomp_features=None, vessel_label=None):
        window = cls.window
        delta = cls.delta
        if label_window is None:
            label_window = delta
        assert window % delta == 0, 'delta must evenly divide window'
        assert label_window % delta == 0, 'delta must evenly divide label_window'
        assert window >= label_window, "window must be at least as large as label_window"
        # Weight so that sets with multiple classification get sqrt(n) more representation
        # Since they have some extra information (n is the number of classifications)
        subsamples = int(round(min_samples / np.sqrt(len(paths))))
        # Set seed for reproducibility
        np.random.seed(seed)
        times = []
        features = []
        transshipping = []
        labels = []
        defined = []
        window_pts = window // delta
        lbl_pts = label_window // delta
        lbl_offset = (window_pts - lbl_pts) // 2
        min_ndx = 1
        for p in paths:
            for data in cls.load_data(p, delta, skip_label, 
                                    keep_fracs=keep_fracs, 
                                    features=precomp_features, 
                                    vessel_label=vessel_label):
                if data is None:
                    print('skipping', p)
                    continue
                (t, x, y, label, dfnd) = data
                
                max_ndx = len(x) - window_pts
                ndxs = []
                for ndx in range(min_ndx, max_ndx + 1):
                    if dfnd[ndx:ndx+window_pts].sum() >= window_pts // 2:
                        ndxs.append(ndx)
                if not ndxs:
                    print("skipping", p, "because it is too short")
                    continue
                for ss in range(subsamples):
                    ndx = np.random.choice(ndxs)                
                    t_chunk = t[ndx:ndx+window_pts]
                    f_chunk, _ = cls.cook_features(y[ndx:ndx+window_pts])
                    times.append(t_chunk) 
                    features.append(f_chunk)
                    if skip_label:
                        transshipping.append(None)
                        labels.append(None)
                        defined.append(None)
                    else:
                        transshipping.append(label[ndx:ndx+window_pts]) 
                        windowed_labels = label[ndx+lbl_offset:ndx+lbl_offset+lbl_pts]
                        labels.append(windowed_labels.mean() > 0.5)
                        windowed_defined = dfnd[ndx+lbl_offset:ndx+lbl_offset+lbl_pts]
                        defined.append(windowed_defined.mean() > 0.5)
        return times, np.array(features), np.array(labels), np.array(transshipping), np.array(defined)  ### CHANGE




  






    


