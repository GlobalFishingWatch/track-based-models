import datetime
import numpy as np
import os
import pandas as pd
from . import util
from .util import minute, lin_interp, interp_degrees 
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

    def create_features_and_times(self, data, angle=77, max_deltas=0):
        t, xi, y, label_i, defined_i = self.build_features(data, skip_label=True)
        min_ndx = 0
        max_ndx = len(y) - self.time_points
        features = []
        times = []
        i0 = 0
        while i0 <= max_ndx:
            i1 = min(i0 + self.time_points + max_deltas * self.time_point_delta, len(y))
            raw_features = y[i0:i1]
            features.append(self.cook_features(raw_features, angle=angle, noise=0)[0])
            i0 = i0 + max_deltas * self.time_point_delta + 1
        times = t[self.time_points//2:-self.time_points//2]
        return features, times

    @classmethod
    def build_features(cls, obj, skip_label=False, keep_frac=1.0):
        # TODO: compute t, dt and xp once and then use simple
        # TODO: interp
        # TODO: rework interp_angle to also use this.
        delta = cls.delta
        n_pts = len(obj['lat'])
        if n_pts == 0:
            return [], [], [], [], []
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

        xi, speeds = lin_interp(obj, 'speed', delta=delta, mask=mask)
        _, courses = interp_degrees(obj, 'course', delta=delta, mask=mask)
        _, lats = lin_interp(obj, 'lat', delta=delta, mask=mask)
        _, lons = interp_degrees(obj, 'lon', delta=delta, mask=mask)
        # Compute the interval between the interpolated point and the nearest 
        # point in the base data. The objective is to give the model an idea
        # how reliable a point is.
        xp = util.compute_xp(obj, mask)
        dts = util.delta_times(xi, xp)
        # If the base data is already interpolated, add the intervals
        if 'min_dt_min' in obj:
            _, extra_mins= lin_interp(obj, 'min_dt_min', delta=delta, mask=mask)
            dts += extra_mins * 60
        _, elevs = lin_interp(obj, 'elevation', delta=delta, mask=mask)
        _, dists = lin_interp(obj, 'distance', delta=delta, mask=mask)
        # Times
        y = np.transpose([speeds, 
                          courses,
                          lats, 
                          lons, 
                          dts, 
                          elevs, 
                          dists])

        t0 = obj['timestamp'].iloc[0]
        t = [(t0 + datetime.timedelta(seconds=delta * i)) for i in range(len(y))]
        #
        if skip_label:
            label = defined = None
        else:
             # Don't mask labels - use non dropped labels for training 
            _, raw_label = lin_interp(obj, cls.data_source_lbl, delta=delta, mask=None,
                                func=lambda v: [x in cls.data_true_vals for x in v])
            # Quick and dirty nearest neighbor (must be binary label)
            label = (raw_label > 0.5)
            _, raw_defined = lin_interp(obj, cls.data_source_lbl, delta=delta, mask=None,
                                func=lambda v: [x in cls.data_defined_vals for x in v])
            defined = (raw_defined > 0.5)
        return np.asarray(t), xi, y, label, defined

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None):
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 2*np.pi) if (angle is None) else angle
        angle_feat = angle + (np.pi / 2.0 - raw_features[:, 1])
        
        ndx = np.random.randint(len(raw_features))
        lat0 = raw_features[ndx, 2]
        lon0 = raw_features[ndx, 3]
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        scale = np.cos(np.radians(lat))
        d1 = lat - lat0
        d2 = (lon - lon0) * scale
        dir_a = np.cos(angle) * d2 - np.sin(angle) * d1
        dir_b = np.cos(angle) * d1 + np.sin(angle) * d2
        depth = -raw_features[:, 5]
        distance = raw_features[:, 6]

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        noisy_time = np.maximum(raw_features[:, 4] / 
                                float(cls.data_far_time) + noise, 0)
        is_far = np.exp(-noisy_time) 
        return np.transpose([speed,
                             np.cos(angle_feat), 
                             np.sin(angle_feat),
                             dir_a,
                             dir_b,
                             0 * is_far,
                             depth, 
                             ]), angle

    @classmethod
    def merge_train_with_features(cls, ssvid, train, features):
         # Filter features down to just the ssvid / time span we want
        mask = (features.ssvid == ssvid)
        features = features[mask]
        features = features.sort_values(by='timestamp')
        # TODO: parameterize
        padding = datetime.timedelta(hours=6)
        t0 = train['timestamp'].iloc[0] - padding
        t1 = train['timestamp'].iloc[-1] + padding
        i0 = np.searchsorted(features.timestamp, t0, side='left')
        i1 = np.searchsorted(features.timestamp, t1, side='right')
        features = features.iloc[i0:i1]
        cls.add_obj_data(train, features)
        # TODO: revamp so we use features name directly
        # Rename so we can use features as obj:
        mapping = {
            'timestamp' : features.timestamp,
            'speed' : features.speed_knots,
            'course' : features.course_degrees,
            'lat' : features.lat,
            'lon' : features.lon,
            'elevation' : features.elevation_m,
            'distance' : features.distance_from_shore_km,
            cls.data_source_lbl : features[cls.data_source_lbl],
            }
        if 'min_dt_min' in features.columns:
            mapping['min_dt_min'] = features.min_dt_min
        return pd.DataFrame(mapping)

    @classmethod
    def load_data(cls, path, features):
        ssvid, train = util.load_training_data(path, cls.vessel_label, cls.data_source_lbl)
        return cls.merge_train_with_features(ssvid, train, features)


    @classmethod
    def add_obj_data(cls, obj, features):
        obj[cls.data_target_lbl] = [(i in cls.data_true_vals) for i in obj[cls.data_source_lbl]]
        _, raw_is_true = lin_interp(obj, cls.data_target_lbl, t=features.timestamp, 
                                    mask=None, # Don't mask labels - use undropped labels for training 
                                    func=lambda x: np.array(x) == 1) # is it a set
        is_true_mask = raw_is_true > 0.5

        obj['is_defined'] = [(i in cls.data_defined_vals) for i in obj[cls.data_source_lbl]]
        _, raw_is_defined = lin_interp(obj, 'is_defined', t=features.timestamp, 
                                      mask=None, # Don't mask labels - use undropped labels for training 
                                      func=lambda x: np.array(x) == 1) # is it a set
        is_defined_mask = raw_is_defined > 0.5

        source = []
        for is_defined, is_true in zip(is_defined_mask, is_true_mask):
            if is_defined:
                if is_true:
                    source.append(cls.data_true_vals[0])
                else:
                    source.append(cls.data_false_vals[0])
            else:
                source.append(cls.data_undefined_vals[0])
        features[cls.data_source_lbl] = source


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
        lbl_pts = label_window // delta
        lbl_offset = (window_pts - lbl_pts) // 2
        min_ndx = 0
        for data in src_objs:
            for kf in keep_fracs:
                t, x, y, label, dfnd = cls.build_features(data, skip_label=skip_label, keep_frac=kf)
                
                max_ndx = len(x) - window_pts
                ndxs = []
                for ndx in range(min_ndx, max_ndx + 1):
                    if dfnd[ndx+lbl_offset:ndx+lbl_offset+lbl_pts].sum() >= lbl_pts / 2.0:
                        ndxs.append(ndx)
                if not ndxs:
                    print("skipping", p, "because it is too short")
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

    def preprocess(self, x, fit=False):
        x0 = np.asarray(x) 
        try:
            x = 0.5 * (x0[:, 1:, :] + x0[:, :-1, :])
            x[:, :, 3:5] = x0[:, 1:, 3:5] - x0[:, :-1, 3:5]
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
