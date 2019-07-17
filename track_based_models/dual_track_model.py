import datetime
import numpy as np
import os
import pandas as pd
from . import util
from .util import minute, lin_interp, cos_deg, sin_deg 
from .base_model import BaseModel

class DualTrackModel(BaseModel):

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

    data_far_time = None

    #TODO: can this be refactored to BaseModel
    @classmethod
    def create_features_and_times(cls, data, angle=77, max_deltas=0):
        t, xi, y, label_i, defined_i = cls.build_features(data, skip_label=True)
        min_ndx = 0
        max_ndx = len(y) - cls.time_points
        features = []
        times = []
        i0 = 0
        while i0 < max_ndx:
            i1 = min(i0 + cls.time_points + max_deltas * cls.time_point_delta, len(y))
            raw_features = y[i0:i1]
            features.append(cls.cook_features(raw_features, angle=angle, noise=0)[0])
            i0 = i0 + max_deltas * cls.time_point_delta + 1
        times = t[cls.time_points//2:-cls.time_points//2]
        return features, times

    @classmethod
    def build_features(cls, obj, skip_label=False, keep_frac=1.0, interp_t=None):
        n_pts = len(obj['timestamp'])
        assert 0 < keep_frac <= 1
        if keep_frac == 1:
            mask = None
        else:           
            # Build a random mask with probability keep_frac. Force
            # first and last point to be true so the time frame
            # stays the same.
            mask = np.random.uniform(0, 1, size=[n_pts]) < keep_frac
            mask[0] = mask[-1] = True 
            
        delta = None if (interp_t is not None) else cls.delta

        assert np.isnan(obj['speed']).sum() == np.isnan(obj['course']).sum() == 0 

        v = np.array(obj['speed'])
        # Replace missing speeds with arbitrary 3.5 (between setting and hauling)
        # TODO: use implied speed instead
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
        # Times
        t0 = obj['timestamp'][0]
        if interp_t is None:
            t = [(t0 + datetime.timedelta(seconds=cls.delta * i)) for i in range(len(y1))]
        else:
            t = interp_t
        y = np.transpose([y0, y1, y2, y3, y4])
        #
        # Quick and dirty nearest neighbor (only works for binary labels I think)
        if skip_label:
            label_i = defined_i = None
        else:
            obj['is_defined'] = [0 if i == 0 else 1 for i in obj['transshiping']]
            obj['is_target_encounter'] = [1 if i == 1 else 0 for i in obj['transshiping']]
            _, raw_label_i = lin_interp(obj, 'is_target_encounter', delta=delta, t=interp_t, mask=None, # Don't mask labels - use undropped labels for training 
                                        func=lambda x: np.array(x) == 1) # is it a set
            label_i = raw_label_i > 0.5

            _, raw_defined_i = lin_interp(obj, 'is_defined', delta=delta, t=interp_t, mask=None, # Don't mask labels - use undropped labels for training 
                                        func=lambda x: np.array(x) == 1) # is it a set
            defined_i = raw_defined_i > 0.5
        #
        return t, xi, y, label_i, defined_i

    @classmethod
    def cook_features(cls, raw_features, angle=None, noise=None,
                            lat0=None, lon0=None):
        speed = raw_features[:, 0]
        angle = np.random.uniform(0, 2*np.pi) if (angle is None) else angle
        angle_feat = angle + (np.pi / 2.0 - raw_features[:, 1])
        
        lat = raw_features[:, 2] 
        lon = raw_features[:, 3] 
        if lat0 is None:
            which = np.random.randint(len(raw_features))
            lat0 = lat[which]
        if lon0 is None:
            which = np.random.randint(len(raw_features))
            lon0 = lon[which]
        scale = np.cos(np.radians(lat))
        d1 = lat - lat0
        d2 = (lon - lon0) * scale
        dir_a = np.cos(angle) * d2 - np.sin(angle) * d1
        dir_b = np.cos(angle) * d1 + np.sin(angle) * d2

        if noise is None:
            noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
        noisy_time = np.maximum(raw_features[:, 4] / float(cls.data_far_time) + noise, 0)
        is_far = np.exp(-noisy_time) 
        return np.transpose([speed, 
                          np.cos(angle_feat), 
                          np.sin(angle_feat),
                          dir_a,
                          dir_b,
                          is_far
                          ]), angle, lat0, lon0

    @classmethod
    def load_obj(cls, path, skip_label=False, keep_fracs=[1], features=None,
                         vessel_label=None, suffix='', extra_labels=()):
        obj_tv = util.load_json_data(path, vessel_label=vessel_label)  
        obj = util.convert_from_legacy_format(obj_tv)
        for lbl in  extra_labels:   
            obj[lbl] = obj_tv[lbl]
        # if features is None:
        if features is not None:
            # Filter features down to just the ssvid / time span we want
            ssvid = os.path.basename(path).split('_')[0]
            mask = (features['id' + suffix] == ssvid)
            features = features[mask]
            features = features.sort_values(by='timestamp')
            t0 = obj['timestamp'].iloc[0]
            t1 = obj['timestamp'].iloc[-1]
            i0 = np.searchsorted(features.timestamp, t0, side='left')
            i1 = np.searchsorted(features.timestamp, t1, side='right')
            features = features.iloc[i0:i1]
            # Add obj data to features
            if extra_labels:
                # TODO: clean up this logic. 
                # Object data adds the true / false data which we only
                # do for one of the branches.
                cls.add_obj_data(obj, features)
            # Rename so we can use features as obj:
            cols = {
                'timestamp' : features.timestamp,
                'speed' : getattr(features, 'speed_knots' + suffix),
                'course' : getattr(features, 'course_degrees' + suffix),
                'lat' : getattr(features, 'lat' + suffix),
                'lon' : getattr(features, 'lon' + suffix),
                }
            for lbl in extra_labels:
                cols[lbl] = features[lbl]
            obj = pd.DataFrame(cols)

        return obj


    @classmethod
    def load_paired_data(cls, path, delta, vessel_labels,
                        skip_label=False, keep_fracs=[1], features=None):
        obj_tv = cls.load_obj(path, vessel_label=vessel_labels[0], suffix='_1',
                            features=features, extra_labels=[cls.data_source_lbl])  
        obj_fv = cls.load_obj(path, vessel_label=vessel_labels[1], suffix='_2',
                              features=features)

        for kf in keep_fracs:
            try:
                t, x, y_tv, label, is_defined = cls.build_features(obj_tv,  
                                                skip_label=skip_label, keep_frac=kf)
                t_fv, x_fv, y_fv, _, _ = cls.build_features(obj_fv, interp_t = t, 
                                                 skip_label=True, keep_frac=kf)
                assert np.all(t_fv == t)
        
                t = np.asarray(t)

                yield (t, x, y_tv, x_fv, y_fv, label, is_defined)
            except:
                raise
                print('skipping', path, kf)
                continue





    @classmethod
    def add_obj_data(cls, obj, features):
        obj['is_defined'] = [(i in cls.data_defined_vals) for i in obj[cls.data_source_lbl]]
        obj[cls.data_target_lbl] = [(i in cls.data_true_vals) for i in obj[cls.data_source_lbl]]
        _, raw_label_i = lin_interp(obj, cls.data_target_lbl, t=features.timestamp, 
                                    mask=None, # Don't mask labels - use undropped labels for training 
                                    func=lambda x: np.array(x) == 1) # is it a set
        features[cls.data_target_lbl] = raw_label_i > 0.5

        _, raw_defined_i = lin_interp(obj, 'is_defined', t=features.timestamp, 
                                      mask=None, # Don't mask labels - use undropped labels for training 
                                      func=lambda x: np.array(x) == 1) # is it a set
        features['is_defined'] = raw_defined_i > 0.5
        source = []
        for is_def, is_true in zip(features['is_defined'],
                                   features[cls.data_target_lbl]):
            if is_def:
                if is_true:
                    source.append(cls.data_true_vals[0])
                else:
                    source.append(cls.data_false_vals[0])
            else:
                source.append(cls.data_undefined_vals[0])
        features[cls.data_source_lbl] = source

    @classmethod
    def generate_data(cls, paths, min_samples, seed=888, 
                    skip_label=False, keep_fracs=(1,), noise=None, 
                    precomp_features=None, vessel_label=None,
                    extra_time_deltas=0):
        delta = cls.delta
        window = cls.window + extra_time_deltas * cls.time_point_delta * delta
        label_window = delta * (1 + extra_time_deltas * cls.time_point_delta)
        assert window % delta == 0, 'delta must evenly divide window'
        # Weight so that sets with multiple classification get sqrt(n) more representation
        # Since they have some extra information (n is the number of classifications)
        subsamples = int(round(min_samples / np.sqrt(len(paths))))
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
        for p in paths:
            for data in cls.load_paired_data(p, delta, 
                                    skip_label=skip_label, 
                                    keep_fracs=keep_fracs, 
                                    features=precomp_features, 
                                    # TODO: make labels into class attributes
                                    vessel_labels=['position_data_reefer',
                                                   'position_data_fishing_vessel']):
                if data is None:
                    print('skipping', p)
                    continue
                (t, x, y, x_fv, y_fv, label, dfnd) = data
                
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
                    _, f_chunk = cls.cook_paired_data(*data, noise=noise,
                                    start_ndx=ndx, end_ndx=ndx+window_pts)
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
        return np.array(times), np.array(features), np.array(labels), np.array(targets), np.array(defined)  ### CHANGE

    @classmethod
    def cook_paired_data(cls, t, x, y_tv, x_fv, y_fv, label, is_defined, start_ndx=0, end_ndx=None,
                        noise=None):
        t, x, y_tv, x_fv, y_fv, label, is_defined = [v[start_ndx:end_ndx] for v in 
                                                       (t, x, y_tv, x_fv, y_fv, label, is_defined)]
            
        features_tv, angle, lat0, lon0 = cls.cook_features(y_tv, noise=noise)
        features_fv, _, _, _ = cls.cook_features(y_fv, angle=angle,
                                                    lat0=lat0, 
                                                    lon0=lon0,
                                                    noise=noise)

        # Relative speed
        f_speed = np.hypot(features_tv[:, 0] * features_tv[:, 1] - features_fv[:, 0] * features_fv[:, 1],
                           features_tv[:, 0] * features_tv[:, 2] - features_fv[:, 0] * features_fv[:, 2])
        f_tspeed = np.log(0.1 + features_tv[:, 0])
        f_dist = np.hypot(features_fv[:, 3] - features_tv[:, 3], features_fv[:, 4] - features_tv[:, 4])
        fv_isfar = features_fv[:, -1]
        
        features = np.concatenate([features_tv[:, :], features_fv[:, :],
                                   f_speed[:,None], f_dist[:, None]], axis=1)
        return t, features

# 

