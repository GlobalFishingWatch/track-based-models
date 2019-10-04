from collections import namedtuple
import datetime
import logging
import numpy as np
import os
import pandas as pd
from . import util
from .util import minute
from .base_model import BaseModel, Normalizer

class SingleTrackModel(BaseModel):

    delta = None
    time_points = None
    time_point_delta = None
    window = None

    data_source_lbl = None 
    data_target_lbl = None
    data_undefined_vals = None
    data_defined_vals = None
    data_true_vals = None
    data_false_vals = None

    vessel_label = None

    feature_padding_hours = 6.0


    def create_features_and_times(self, data, angle=77, max_deltas=0):
        t, y, _, _ = self.build_features(data, skip_label=True)
        # First get all large chunks
        max_ndx = len(y) - (self.time_points + max_deltas * self.time_point_delta)
        features = []
        times = []
        i0 = 0
        while i0 <= max_ndx:
            i1 = i0 + self.time_points + max_deltas * self.time_point_delta
            features.append(self.cook_features(y[i0:i1], angle=angle, noise=0)[0])
            i0 = i0 + max_deltas * self.time_point_delta + 1
        # Now get all small chunks
        max_ndx = len(y) - self.time_points
        while i0 <= max_ndx:
            i1 = i0 + self.time_points
            features.append(self.cook_features(y[i0:i1], angle=angle, noise=0)[0])
            i0 = i0 + 1
        times = t[self.time_points//2:-(self.time_points//2)]
        return features, times


    @classmethod
    def make_mask(cls, n_pts, keep_frac):
        assert 0 < keep_frac <= 1, 'keep frac must be between 0 and 1'
        if keep_frac == 1:
            mask = None
        else:           
            # Build a random mask with probability keep_frac. Force
            # first and last point to be true so the time frame
            # stays the same.
            mask = np.random.uniform(0, 1, size=[n_pts]) < keep_frac
            mask[0] = mask[-1] = True 
        return mask

    @classmethod
    def build_features(cls, obj, skip_label=False, keep_frac=1.0):
 
        assert (np.isnan(obj.speed_knots).sum() == 
                np.isnan(obj.course_degrees).sum() == 0), (
                'null values are not allow in the data, please filter first')
 
        n_pts = len(obj['lat'])
        if n_pts == 0:
            return [], [], [], []

        mask = cls.make_mask(n_pts, keep_frac)
        interp_info = util.setup_lin_interp(obj, delta=cls.delta, mask=mask)

        speeds = util.do_lin_interp(obj, interp_info, 'speed_knots')
        courses = util.do_degree_interp(obj, interp_info, 'course_degrees')
        lats = util.do_degree_interp(obj, interp_info, 'lat')
        lons = util.do_degree_interp(obj, interp_info, 'lon')
        elevs = util.do_lin_interp(obj, interp_info, 'elevation_m')
        dists = util.do_lin_interp(obj, interp_info, 'distance_from_shore_km')

        # Compute the interval between the interpolated point and the nearest 
        # point in the base data. The objective is to give the model an idea
        # how reliable a point is.
        dts = util.delta_times(interp_info.interp_seconds, interp_info.raw_seconds)
        # If the base data is already interpolated, add the intervals
        if 'min_dt_min' in obj:
            dts += util.do_lin_interp(obj, interp_info, 'min_dt_min') * 60

        y = np.transpose([speeds, courses, lats, lons, 
                          dts, elevs, dists])

        if skip_label:
            label = defined = None
        else:
             # Don't mask labels - use non dropped labels for training 
            unmasked_interp_info = util.setup_lin_interp(obj, delta=cls.delta)
            label = util.do_lin_interp(obj, unmasked_interp_info, cls.data_source_lbl,
                                func=lambda v: [x in cls.data_true_vals for x in v]) > 0.5
            defined = util.do_lin_interp(obj, unmasked_interp_info, cls.data_source_lbl,
                                func=lambda v: [x in cls.data_defined_vals for x in v]) > 0.5

        return interp_info.interp_timestamps, y, label, defined


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
        latavg = 0.5 * (lat[1:] + lat[:-1])
        scale = np.cos(np.radians(latavg))
        d1 = lat[1:] - lat[:-1]
        d2 = ((lon[1:] - lon[:-1] + 180) % 360 - 180) * scale
        dir_a = np.cos(radians) * d2 - np.sin(radians) * d1
        dir_b = np.cos(radians) * d1 + np.sin(radians) * d2
        dir_a = np.concatenate([dir_a[:1], dir_a], axis=0)
        dir_b = np.concatenate([dir_b[:1], dir_b], axis=0)
        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

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

        return np.transpose([f.speed,
                             f.speed * np.cos(np.radians(f.angle_feature)), 
                             f.speed * np.sin(np.radians(f.angle_feature)),
                             f.dir_a,
                             f.dir_b,
                             np.exp(-f.delta_time),
                             logged_depth, 
                             ]), angle

    @classmethod
    def merge_train_with_features(cls, ssvid, train, features):
         # Filter features down to just the ssvid / time span we want
        mask = (features.ssvid == ssvid)
        features = features[mask]
        features = features.sort_values(by='timestamp')
        padding = datetime.timedelta(hours=cls.feature_padding_hours)
        t0 = train['timestamp'].iloc[0] - padding
        t1 = train['timestamp'].iloc[-1] + padding
        i0 = np.searchsorted(features.timestamp, t0, side='left')
        i1 = np.searchsorted(features.timestamp, t1, side='right')
        features = features.iloc[i0:i1]
        cls.add_obj_data(train, features)
        return features

    @classmethod
    def load_data(cls, path, features):
        ssvid, train = util.load_training_data(path, cls.vessel_label, cls.data_source_lbl)
        return cls.merge_train_with_features(ssvid, train, features)

    @classmethod
    def add_obj_data(cls, obj, features):
         # Don't mask labels - use non dropped labels for training 
        unmasked_interp_info = util.setup_lin_interp(obj, timestamps=features.timestamp)
        is_true_mask = util.do_lin_interp(obj, unmasked_interp_info, cls.data_source_lbl,
                            func=lambda v: [x in cls.data_true_vals for x in v]) > 0.5
        is_defined_mask = util.do_lin_interp(obj, unmasked_interp_info, cls.data_source_lbl,
                            func=lambda v: [x in cls.data_defined_vals for x in v]) > 0.5

        sources = []
        for is_defined, is_true in zip(is_defined_mask, is_true_mask):
            if is_defined:
                if is_true:
                    src = cls.data_true_vals[0]
                else:
                    src = cls.data_false_vals[0]
            else:
                src = cls.data_undefined_vals[0]
            sources.append(src)

        features[cls.data_source_lbl] = sources


    @classmethod
    def generate_inputs(cls, src_objs, min_samples, seed=888, 
                    skip_label=False, keep_fracs=(1,), noise=None, extra_time_deltas=0):
        delta = cls.delta
        window = cls.window + extra_time_deltas * cls.time_point_delta * delta
        label_window = delta * (1 + extra_time_deltas * cls.time_point_delta)
        assert window % delta == 0, 'delta must evenly divide window'
        # Weight so that sets with multiple classification get sqrt(n) more representation
        # Since they have some extra information (n is the number of classifications)
        subsamples = int(round(min_samples / np.sqrt(len(src_objs))))
        # Set seed for reproducibility
        np.random.seed(seed)
        times = []
        features = []
        targets = []
        labels = []
        defined = []
        window_pts = window // delta
        assert window_pts == cls.time_points + extra_time_deltas * cls.time_point_delta
        lbl_pts = label_window // delta
        assert lbl_pts == 1 + extra_time_deltas * cls.time_point_delta
        lbl_offset = (window_pts - lbl_pts) // 2
        min_ndx = 0
        for i, data in enumerate(src_objs):
            for kf in keep_fracs:
                t, y, label, dfnd = cls.build_features(data, skip_label=skip_label, keep_frac=kf)
                
                max_ndx = len(y) - window_pts
                ndxs = []
                for ndx in range(min_ndx, max_ndx + 1):
                    if dfnd[ndx+lbl_offset:ndx+lbl_offset+lbl_pts].sum() >= lbl_pts / 2.0:
                        ndxs.append(ndx)
                if not ndxs:
                    print("skipping object", i, "because it is too short")
                    print(len(dfnd), np.sum(dfnd), 
                        sorted(set(label)))
                    continue
                for ss in range(subsamples):
                    ndx = np.random.choice(ndxs)                
                    t_chunk = t[ndx:ndx+window_pts]
                    f_chunk, _ = cls.cook_features(y[ndx:ndx+window_pts], noise=noise)
                    times.append(t_chunk) 
                    features.append(f_chunk)
                    if skip_label:
                        targets.append(None)
                        labels.append(None)
                        defined.append(None)
                    else:
                        # print(label[ndx+lbl_offset:ndx+lbl_offset+lbl_pts].shape, 
                        #     lbl_pts)
                        targets.append(label[ndx:ndx+window_pts]) 
                        windowed_labels = label[ndx+lbl_offset:ndx+lbl_offset+lbl_pts].reshape(
                            lbl_pts, -1)
                        labels.append(windowed_labels.mean(axis=-1) > 0.5)
                        windowed_defined = dfnd[ndx+lbl_offset:ndx+lbl_offset+lbl_pts].reshape(
                            lbl_pts, -1)
                        defined.append((windowed_defined.mean(axis=-1) > 0.5) &
                            ((windowed_labels.mean(axis=-1) < 0.3) | 
                             (windowed_labels.mean(axis=-1) > 0.7)))
        return times, np.array(features), np.array(labels), np.array(targets), np.array(defined) 


class SingleTrackDiffModel(SingleTrackModel):
    """SingleTrackModel that uses delta lat/lon
    """

    def create_features_and_times(self, data, angle=77, max_deltas=0):
        f, t = SingleTrackModel.create_features_and_times(self, data, angle, max_deltas)
        if len(t):
            t = t + datetime.timedelta(seconds=self.delta / 2.0)
        return f, t

    def preprocess(self, x, fit=False):
        x0 = np.asarray(x) 
        try:
            x = 0.5 * (x0[:, 1:, :] + x0[:, :-1, :])
            x[:, :, 3:5] = x0[:, 1:, 3:5]
        except:
            logging.error('x is wrong shape: {}'.format(x0.shape))
            raise
        if fit:
            self.normalizer = Normalizer().fit(x)
        return self.normalizer.norm(x)

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
