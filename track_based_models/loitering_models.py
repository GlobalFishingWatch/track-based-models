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
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    
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
    def build_features(cls, data, skip_label=True, keep_frac=1.0):
        return build_features(data, delta=cls.delta, 
                              skip_label=skip_label, keep_frac=keep_frac)
                    # source_lbl='transshipping', 
                    # target_lbl='is_target_encounter',
                    # defined_vals = [1, 2], true_vals = [1])


    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None, 
                                            far_time=3 * 10 * minute):
        return cook_features(raw_features, angle, noise, far_time)



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
            for data in load_data(p, delta, skip_label, 
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
                    f_chunk, _ = cook_features(y[ndx:ndx+window_pts])
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




  
def add_obj_data(obj, features):
    obj['is_defined'] = [0 if i in (0, 3) else 1 for i in obj['transshiping']]
    obj['is_target_encounter'] = [1 if i == 1 else 0 for i in obj['transshiping']]
    _, raw_label_i = lin_interp(obj, 'is_target_encounter', t=features.timestamp, 
                                mask=None, # Don't mask labels - use undropped labels for training 
                                func=lambda x: np.array(x) == 1) # is it a set
    features['is_target_encounter'] = raw_label_i > 0.5

    _, raw_defined_i = lin_interp(obj, 'is_defined', t=features.timestamp, 
                                  mask=None, # Don't mask labels - use undropped labels for training 
                                  func=lambda x: np.array(x) == 1) # is it a set
    features['is_defined'] = raw_defined_i > 0.5
    # Map not defined to 0, is_target to 1, and not_target to 2
    features['transshiping'] = features['is_defined'] * (
                                2 - features['is_target_encounter'])


def cook_features(raw_features, angle=None, noise=None, far_time=3 * 10 * minute):
    speed = raw_features[:, 0]
    angle = np.random.uniform(0, 2*np.pi) if (angle is None) else angle
    angle_feat = angle + (np.pi / 2.0 - raw_features[:, 1])
    
    ndx = len(raw_features) // 2
    lat0 = raw_features[ndx, 2]
    lon0 = raw_features[ndx, 3]
    lat = raw_features[:, 2] 
    lon = raw_features[:, 3] 
    scale = np.cos(np.radians(lat))
    d1 = lat - lat0
    d2 = (lon - lon0) * scale
    dir_a = np.cos(angle) * d2 - np.sin(angle) * d1
    dir_b = np.cos(angle) * d1 + np.sin(angle) * d2

    if noise is None:
        noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
    noisy_time = np.maximum(raw_features[:, 4] / float(far_time) + noise, 0)
    is_far = np.exp(-noisy_time) 
    dir_h = np.hypot(dir_a, dir_b)
    return np.transpose([speed,
                         np.cos(angle_feat), 
                         np.sin(angle_feat),
                         dir_a,
                         dir_b,
                         is_far
                         ]), angle



def load_data(path, delta, skip_label=False, keep_fracs=[1], features=None,
                     vessel_label=None):
    obj_tv = util.load_json_data(path, vessel_label=vessel_label)  
    obj = util.convert_from_legacy_format(obj_tv)
    obj['transshiping'] = obj_tv['transshiping']
    # if features is None:
    if features is not None:
        # Filter features down to just the ssvid / time span we want
        ssvid = os.path.basename(path).split('_')[0]
        mask = (features.ssvid == ssvid)
        features = features[mask]
        features = features.sort_values(by='timestamp')
        t0 = obj['timestamp'].iloc[0]
        t1 = obj['timestamp'].iloc[-1]
        i0 = np.searchsorted(features.timestamp, t0, side='left')
        i1 = np.searchsorted(features.timestamp, t1, side='right')
        features = features.iloc[i0:i1]
        # Add transshiping data to features
        add_obj_data(obj, features)
        # Rename so we can use features as obj:
        obj = pd.DataFrame({
            'timestamp' : features.timestamp,
            'speed' : features.speed_knots,
            'course' : features.course_degrees,
            'lat' : features.lat,
            'lon' : features.lon,
            'transshiping' : features.transshiping,
            })
    for kf in keep_fracs:
        try:
            t, x, y, label, is_defined = LoiteringModelV1.build_features(obj, delta=delta, 
                                            skip_label=skip_label, keep_frac=kf)
        except:
            print('skipping', path, kf)
            continue
        t = np.asarray(t)
        yield (t, x, y, label, is_defined)
    


