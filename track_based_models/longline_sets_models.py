from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, ELU, ReLU, Input, Conv1D, ZeroPadding1D
from keras.layers import MaxPooling1D, AveragePooling1D, BatchNormalization, Cropping1D
from keras.layers.core import Activation
from keras import optimizers
from .util import hour, minute
from .base_model import hybrid_pool_layer, Normalizer
from .single_track_model import SingleTrackModel, SingleTrackDiffModel, SingleTrackDistModel
from . import util
from .util import minute, lin_interp, cos_deg, sin_deg  
from collections import namedtuple
from .shake_shake import ShakeShake

class LonglineSetsModelV1(SingleTrackModel):
    delta = hour
    window = (29 + 4*6) *  delta
    time_points = window // delta
    base_filter_count = 32
    fc_nodes = 512

    def __init__(self):

        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(self.time_points, 6))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer(y)
        
        depth = 2 * depth 
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer(y)
        
        y = Flatten()(y)
        y = Dense(self.fc_nodes)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)

        y = Dense(self.fc_nodes)(y)
        y = ELU()(y)
        y = Dropout(0.5)(y)

        y = Dense(1)(y)
        y = Activation('sigmoid')(y)
        y = Reshape
        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer)
        opt = optimizers.Nadam(lr=0.01)
        #opt = optimizers.Adam(lr=0.01, decay=0.5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  

    @classmethod
    def build_features(cls, data, skip_label=True):
        return build_features(data, delta=cls.delta, skip_label=skip_label)


    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None, 
                                            far_time=3 * 10 * minute):
        return cook_features(raw_features, angle, noise, far_time)


    def predict(self, x):
        x1 = self.preprocess(x)
        return self.model.predict(x1)


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
                    print("skipping", p, "because it is too short", 
                        p, len(x), window_pts)
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



def build_features(obj, delta=None, interp_t=None, 
                    skip_label=False, keep_frac=1.0):
    n_pts = len(obj['lat'])
    assert 0 < keep_frac <= 1, 'keep frac must be between 0 and 1'
    if keep_frac == 1:
        mask = None
    else:           
        # Build a random mask with probability keep_frac. Force
        # first and last point to be true so the time frame
        # stays the same.
        mask = np.random.uniform(0, 1, size=[n_pts]) < keep_frac
        mask[0] = mask[-1] = True 
        
    assert np.isnan(obj['speed']).sum() == np.isnan(obj['course']).sum() == 0, (
            'null values are not allow in the data, please filter first')

    v = np.array(obj['speed'])
    # Replace missing speeds with arbitrary 3.5 (between setting and hauling)
    v[np.isnan(v)] = 3.5
    obj['speed'] = v
    xi, speeds = lin_interp(obj, 'speed', delta=delta, t=interp_t, mask=mask)
    y0 = speeds
    #
    _, cos_yi = lin_interp(obj, 'course', delta=delta, t=interp_t, mask=mask, func=cos_deg)
    _, sin_yi = lin_interp(obj, 'course', delta=delta, t=interp_t, mask=mask, func=sin_deg)
    angle_i = np.arctan2(sin_yi, cos_yi)
    y1 = angle_i
    #
    _, y2 = lin_interp(obj, 'lat', delta=delta, t=interp_t, mask=mask)
    # Longitude can cross the dateline, so interpolate useing cos / sin
    _, cos_yi = lin_interp(obj, 'lon', delta=delta, t=interp_t, mask=mask, func=cos_deg)
    _, sin_yi = lin_interp(obj, 'lon', delta=delta, t=interp_t, mask=mask, func=sin_deg)
    y3 = np.degrees(np.arctan2(sin_yi, cos_yi))
    # delta times
    xp = util.compute_xp(obj, mask)
    dts = util.delta_times(xi, xp)
    y4 = dts
    if 'min_dt_min' in obj:
        dts += lin_interp(obj, 'min_dt_min', t=interp_t, mask=None) * 60
    # Times
    t0 = obj['timestamp'].iloc[0]
    if interp_t is None:
        t = [(t0 + datetime.timedelta(seconds=delta * i)) for i in range(len(y1))]
    else:
        t = interp_t
    y = np.transpose([y0, y1, y2, y3, y4])
    #
    # Quick and dirty nearest neighbor (only works for binary labels I think)
    if skip_label:
        label_i = defined_i = None
    else:
        obj['is_defined'] = [0 if i == 0 else 1 for i in obj['fishing']]
        _, raw_label_i = lin_interp(obj, 'fishing', delta=delta, t=interp_t, 
                                    mask=None, # Don't mask labels - use undropped labels for training 
                                    func=lambda x: np.array(x) == 1) # is it a set
        label_i = raw_label_i > 0.5

        _, raw_defined_i = lin_interp(obj, 'is_defined', delta=delta, t=interp_t, 
                                      mask=None, # Don't mask labels - use undropped labels for training 
                                      func=lambda x: np.array(x) == 1) # is it a set
        defined_i = raw_defined_i > 0.5
    #
    return t, xi, y, label_i, defined_i
  
def add_obj_data(obj, features):
    _, raw_label_i = lin_interp(obj, 'fishing', t=features.timestamp, 
                                mask=None, # Don't mask labels - use undropped labels for training 
                                func=lambda x: np.array(x) == 1) # is it a set
    features['is_fishing'] = raw_label_i > 0.5

    _, raw_defined_i = lin_interp(obj, 'fishing', t=features.timestamp, 
                                  mask=None, # Don't mask labels - use undropped labels for training 
                                  func=lambda x: np.array(x) != 0) # is it a set
    features['is_defined'] = raw_defined_i > 0.5
    features['fishing'] = features['is_defined'] * (2 - features['is_fishing'])


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
    obj['fishing'] = obj_tv['fishing']
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
        # Add fishing data to features
        add_obj_data(obj, features)
        # Rename so we can use features as obj:
        obj = pd.DataFrame({
            'timestamp' : features.timestamp,
            'speed' : features.speed_knots,
            'course' : features.course_degrees,
            'lat' : features.lat,
            'lon' : features.lon,
            'fishing' : features.fishing,
            })
    for kf in keep_fracs:
        try:
            t, x, y, label, is_defined = build_features(obj, delta=delta, 
                                            skip_label=skip_label, keep_frac=kf)
        except:
            print(path, kf, 'failed, skipping')
            continue
        t = np.asarray(t)
        yield (t, x, y, label, is_defined)
    

from .loitering_models import LoiteringModelVD2 as ModelBase

class LonglineSetsModelV2(ModelBase):
    delta = 20 * minute
    time_points = 141 # 144 = 12 hours,
    internal_time_points = 139
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 128
    
    base_filter_count = 8
    
    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (1, 2, 3)
    data_true_vals = (1,)
    data_false_vals = (2, 3)
    
    feature_padding_hours = 24.0



class LonglineSetsModelV3(SingleTrackModel):
    
    delta = hour
    window = (29 + 4*6) *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1
    fc_nodes = 512

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization()(y)
        y1 = MaxPooling1D(pool_size=pool_width, strides=1)(y)
        y2 = AveragePooling1D(pool_size=pool_width, strides=1)(y)
        y = keras.layers.concatenate([y1, y2])

        pool_width  = pool_width * 2 + 1
        y = Conv1D(2 * depth, 3, dilation_rate=2)(y)
        y = ELU()(y)
        y = BatchNormalization()(y)
        y = Conv1D(2 * depth, 3, dilation_rate=2)(y)
        y = ELU()(y)
        y = BatchNormalization()(y)
        y1 = MaxPooling1D(pool_size=pool_width, strides=1)(y)
        y2 = AveragePooling1D(pool_size=pool_width, strides=1)(y)
        y = keras.layers.concatenate([y1, y2])

        y = Conv1D(self.fc_nodes, 9, dilation_rate=4)(y)
        y = ELU()(y)
        y = BatchNormalization()(y)

        y = Conv1D(self.fc_nodes, 1)(y)
        y = ELU()(y)
        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)



class LonglineSetsModelV4(SingleTrackModel):

    custom_objects = {'ShakeShake': ShakeShake}
    
    delta = hour
    window = (29 + 4*6) *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1
    fc_nodes = 512

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y1 = Conv1D(depth, 3)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(depth, 3)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = Conv1D(depth, 3)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(depth, 3)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = MaxPooling1D(pool_size=pool_width, strides=1)(y)
        y2 = AveragePooling1D(pool_size=pool_width, strides=1)(y)
        y = keras.layers.concatenate([y1, y2])

        pool_width  = pool_width * 2 + 1
        y1 = Conv1D(2 * depth, 3, dilation_rate=2)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(2 * depth, 3, dilation_rate=2)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = Conv1D(2 * depth, 3, dilation_rate=2)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(2 * depth, 3, dilation_rate=2)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = MaxPooling1D(pool_size=pool_width, strides=1)(y)
        y2 = AveragePooling1D(pool_size=pool_width, strides=1)(y)
        y = keras.layers.concatenate([y1, y2])

        y1 = Conv1D(self.fc_nodes, 9, dilation_rate=4)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(self.fc_nodes, 9, dilation_rate=4)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)

        y = Conv1D(self.fc_nodes, 1)(y)
        y = ReLU()(y)
        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)



class LonglineSetsModelV5(SingleTrackModel):

    custom_objects = {'ShakeShake': ShakeShake}
    
    delta = hour
    window = (29 + 4*6) *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1
    fc_nodes = 512

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        pool_width = 3
        dilation = 1

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y1 = Conv1D(10 * depth // 10, 3)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(10 * depth // 10, 3)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = Conv1D(14 * depth // 10, 3)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(14 * depth // 10, 3)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = MaxPooling1D(pool_size=pool_width, strides=1)(y)
        y2 = AveragePooling1D(pool_size=pool_width, strides=1)(y)
        y = keras.layers.concatenate([y1, y2])

        pool_width  = pool_width * 2 + 1
        y1 = Conv1D(18 * depth // 10, 3, dilation_rate=2)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(18 * depth // 10, 3, dilation_rate=2)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = Conv1D(22 * depth // 10, 3, dilation_rate=2)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(22 * depth // 10, 3, dilation_rate=2)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = MaxPooling1D(pool_size=pool_width, strides=1)(y)
        y2 = AveragePooling1D(pool_size=pool_width, strides=1)(y)
        y = keras.layers.concatenate([y1, y2])

        y1 = Conv1D(26 * depth // 10, 3, dilation_rate=4)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(26 * depth // 10, 3, dilation_rate=4)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = Conv1D(30 * depth // 10, 3, dilation_rate=4)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(30 * depth // 10, 3, dilation_rate=4)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = Conv1D(34 * depth // 10, 3, dilation_rate=4)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(34 * depth // 10, 3, dilation_rate=4)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)
        y1 = Conv1D(38 * depth // 10, 3, dilation_rate=4)(y)
        y1 = BatchNormalization()(y1)
        y2 = Conv1D(38 * depth // 10, 3, dilation_rate=4)(y)
        y2 = BatchNormalization()(y2)
        y = ShakeShake()([y1, y2])
        y = ReLU()(y)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)


class LonglineSetsModelV6(SingleTrackModel):

    custom_objects = {'ShakeShake': ShakeShake}
    
    delta = hour
    window = (29 + 24) *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        base_filter_count = self.base_filter_count
        dilation = 1

        def shake_shake_block(y, filter_count, dilation_rate):
            filter_count = int(round(filter_count))
            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y1 = BatchNormalization()(y1)
            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y2 = BatchNormalization()(y2)
            y = ShakeShake()([y1, y2])
            y = ReLU()(y)
            return y

        def pool_block(y, width):
            # Would like to use cropped, but may need more data
            y1 = MaxPooling1D(pool_size=width, strides=1)(y)
            y2 = AveragePooling1D(pool_size=width, strides=1)(y)
            y = keras.layers.concatenate([y1, y2])
            return y

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y = shake_shake_block(y, 1.0 * base_filter_count, 1)
        y = shake_shake_block(y, 1.4 * base_filter_count, 1)
        y = pool_block(y, 3)

        y = shake_shake_block(y, 1.8 * base_filter_count, 2)
        y = shake_shake_block(y, 2.2 * base_filter_count, 2)
        y = pool_block(y, 7)

        y = shake_shake_block(y, 2.6 * base_filter_count, 4)
        y = shake_shake_block(y, 3.0 * base_filter_count, 4) 
        y = shake_shake_block(y, 3.4 * base_filter_count, 4)
        y = shake_shake_block(y, 3.8 * base_filter_count, 4)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)

class LonglineSetsModelV7(SingleTrackModel):

    custom_objects = {'ShakeShake': ShakeShake}
    
    delta = hour
    window = (29 + 28) *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        base_filter_count = self.base_filter_count
        dilation = 1

        def shake_shake_block(y, filter_count, dilation_rate):
            filter_count = int(round(filter_count))
            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y1 = BatchNormalization()(y1)
            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y2 = BatchNormalization()(y2)
            y = ShakeShake()([y1, y2])
            y = ReLU()(y)
            return y

        def pool_block(y, width):
            return MaxPooling1D(pool_size=width, strides=1)(y)

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y = shake_shake_block(y, 1.0 * base_filter_count, 1)
        y0 = y = shake_shake_block(y, 1.4 * base_filter_count, 1)
        y = pool_block(y, 3)

        y = shake_shake_block(y, 1.8 * base_filter_count, 2)
        y1 = y = shake_shake_block(y, 2.2 * base_filter_count, 2)
        y = pool_block(y, 7)

        y = shake_shake_block(y, 2.6 * base_filter_count, 4)
        y = shake_shake_block(y, 3.0 * base_filter_count, 4) 
        y = shake_shake_block(y, 2.6 * base_filter_count, 4)

        y = keras.layers.concatenate([y, Cropping1D(cropping=(15, 15))(y1)])
        y = shake_shake_block(y, 2.2 * base_filter_count, 2)
        y = shake_shake_block(y, 2.2 * base_filter_count, 2)

        y = keras.layers.concatenate([y, Cropping1D(cropping=(24, 24))(y0)])
        y = shake_shake_block(y, 1.4 * base_filter_count, 1)
        y = shake_shake_block(y, 1.0 * base_filter_count, 1)


        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)


class LonglineSetsModelV8(SingleTrackModel):

    custom_objects = {'ShakeShake': ShakeShake}
    
    delta = hour
    window = 57 *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        base_filter_count = self.base_filter_count
        dilation = 1

        def shake_shake_block(y, filter_count, dilation_rate):
            filter_count = int(round(filter_count))

            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y1 = BatchNormalization()(y1)
            y1 = ReLU()(y1)
            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y1)
            y1 = BatchNormalization()(y1)

            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y2 = BatchNormalization()(y2)
            y2 = ReLU()(y2)
            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y2)
            y2 = BatchNormalization()(y2)

            y = ShakeShake()([y1, y2])
            y = ReLU()(y)
            return y

        def pool_block(y, width):
            # Would like to use cropped, but may need more data
            y1 = MaxPooling1D(pool_size=width, strides=1)(y)
            y2 = AveragePooling1D(pool_size=width, strides=1)(y)
            y = keras.layers.concatenate([y1, y2])
            return y

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y = shake_shake_block(y, 1.0 * base_filter_count, 1)

        y = shake_shake_block(y, 1.5 * base_filter_count, 2)

        y = shake_shake_block(y, 2.0 * base_filter_count, 4)
        y = shake_shake_block(y, 2.0 * base_filter_count, 4) 

        y = shake_shake_block(y, 1.5 * base_filter_count, 2)

        y = shake_shake_block(y, 1.0 * base_filter_count, 1)


        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)


class LonglineSetsModelV9(SingleTrackModel):

    custom_objects = {'ShakeShake': ShakeShake}
    
    delta = hour
    window = (29 + 24) *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        base_filter_count = self.base_filter_count
        dilation = 1

        def shake_shake_block(y, filter_count, dilation_rate):
            filter_count = int(round(filter_count))

            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y1 = BatchNormalization()(y1)
            y1 = ReLU()(y1)
            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y1)
            y1 = BatchNormalization()(y1)

            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y)
            y2 = BatchNormalization()(y2)
            y2 = ReLU()(y2)
            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y2)
            y2 = BatchNormalization()(y2)

            y = ShakeShake()([y1, y2])
            y = ReLU()(y)
            return y

        def pool_block(y, width):
            # Would like to use cropped, but may need more data
            y1 = MaxPooling1D(pool_size=width, strides=1)(y)
            y2 = AveragePooling1D(pool_size=width, strides=1)(y)
            y = keras.layers.concatenate([y1, y2])
            return y

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y = shake_shake_block(y, 1.0 * base_filter_count, 1)
        y = pool_block(y, 3)

        y = shake_shake_block(y, 1.8 * base_filter_count, 2)
        y = pool_block(y, 7)

        y = shake_shake_block(y, 2.6 * base_filter_count, 4)
        y = shake_shake_block(y, 3.4 * base_filter_count, 4)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)


class LonglineSetsModelV10(SingleTrackModel):

    custom_objects = {'ShakeShake': ShakeShake}
    
    delta = hour
    window = (29 + 24) *  delta 
    time_points = window // delta # 53
    internal_time_points = time_points
    base_filter_count = 32
    time_point_delta = 1

    vessel_label = None
    data_source_lbl='fishing' 
    data_target_lbl='setting'
    data_undefined_vals = (0,)
    data_defined_vals = (0, 1, 2)
    data_true_vals = (1,)
    data_false_vals = (0, 2)
    data_far_time = 3 * 10 * minute

    feature_padding_hours = 24.0


    def __init__(self, width=None):
        
        self.normalizer = None
        
        base_filter_count = self.base_filter_count
        dilation = 1

        def shake_shake_block(y, filter_count, dilation_rate):
            filter_count = int(round(filter_count))

            yc = BatchNormalization()(y)
            yc = ReLU()(yc)

            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(yc)
            y1 = BatchNormalization()(y1)
            y1 = ReLU()(y1)
            y1 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y1)

            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(yc)
            y2 = BatchNormalization()(y2)
            y2 = ReLU()(y2)
            y2 = Conv1D(filter_count, 3, dilation_rate=dilation_rate)(y2)

            y = ShakeShake()([y1, y2])
            return y

        def pool_block(y, width):
            # Would like to use cropped, but may need more data
            y1 = MaxPooling1D(pool_size=width, strides=1)(y)
            y2 = AveragePooling1D(pool_size=width, strides=1)(y)
            y = keras.layers.concatenate([y1, y2])
            return y

        input_layer = Input(shape=(width, 6)) #19
        y = input_layer
        y = shake_shake_block(y, 1.0 * base_filter_count, 1)
        y = pool_block(y, 3)

        y = shake_shake_block(y, 1.8 * base_filter_count, 2)
        y = pool_block(y, 7)

        y = shake_shake_block(y, 2.6 * base_filter_count, 4)
        y = shake_shake_block(y, 3.4 * base_filter_count, 4)

        y = Dropout(0.5)(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)


        model = KerasModel(inputs=input_layer, outputs=y)
        # opt = optimizers.Nadam()
        opt = optimizers.Adam(lr=0.01)
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
        return np.asarray(x)

    AugmentedFeatures = namedtuple('AugmentedFeatures',
        ['speed', 'delta_time', 'angle_feature', 
        'dir_a', 'dir_b', 'depth', 'distance'])
    @classmethod
    def _augment_features(cls, raw_features, angle=None, noise=None):
        """Perform the augmention portion of cook features"""
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 360) if (angle is None) else angle
        radians = np.radians(angle)
        angle_feat = angle + (90 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        n = len(lat) // 2
        d1 = lat - lat[n]
        d2 = ((lon - lon[n] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2

        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        # speed, angle_feat, depth, distance are all off so shift one

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        delta_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)

        return angle, cls.AugmentedFeatures(speed, delta_time, angle_feat, 
                        dir_a, dir_b, depth, distance)

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        angle, f = cls._augment_features(raw_features, angle, noise)

        if noise is None:
            noise = np.random.normal(0, .05, size=len(f.depth))
        depth = np.maximum(f.depth, 0)
        logged_depth = np.log(1 + depth) + 40 * noise

        dt = cls.delta / hour

        return np.transpose([f.speed,
                             np.cos(np.radians(f.angle_feature)) * dt, 
                             np.sin(np.radians(f.angle_feature)) * dt,
                             f.dir_a * 60,
                             f.dir_b * 60,
                             np.exp(-f.delta_time), 
                             ]), angle

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, 
            validation_split=0, validation_data=0, verbose=1, callbacks=[],
            initial_epoch=0):
        x1 = self.preprocess(x, fit=True)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data,
                      initial_epoch=initial_epoch,
                      verbose=verbose, callbacks=callbacks)