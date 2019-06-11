from __future__ import division
from __future__ import print_function
import datetime
import dateutil.parser
from glob import glob
import json
import numpy as np
import pandas as pd
import os

from . import util
from .util import minute, lin_interp, cos_deg, sin_deg  

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

# def build_features_2(obj, features, 
#                     skip_label=False, keep_frac=1.0):
#     n_pts = len(features['lat'])
#     assert 0 < keep_frac <= 1, 'keep frac must be between 0 and 1'
#     if keep_frac == 1:
#         mask = None
#     else:           
#         # raise ValueError('only keep_frac of 1 allowed') 
#         # Build a random mask with probability keep_frac. Force
#         # first and last point to be true so the time frame
#         # stays the same.
#         mask = np.random.uniform(0, 1, size=[n_pts]) < keep_frac
#         mask[0] = mask[-1] = True 

#     assert np.isnan(features['speed_knots'].values).sum() == np.isnan(features['course_degrees'].values).sum() == 0, (
#             'null values are not allow in the data, please filter first')

#     xi = (features.timestamp - features.timestamp.iloc[0]).dt.total_seconds()
#     interp_t = features.timestamp

#     xi_2, speeds = lin_interp(features, 'speed_knots', t=interp_t, mask=mask)
#     y0 = speeds
#     #
#     _, cos_yi = lin_interp(features, 'course_degrees', t=interp_t, mask=mask, func=cos_deg)
#     _, sin_yi = lin_interp(features, 'course_degrees', t=interp_t, mask=mask, func=sin_deg)
#     angle_i = np.arctan2(sin_yi, cos_yi)
#     y1 = angle_i
#     #
#     _, y2 = lin_interp(features, 'lat', t=interp_t, mask=mask)
#     # Longitude can cross the dateline, so interpolate useing cos / sin
#     _, cos_yi = lin_interp(features, 'lon', t=interp_t, mask=mask, func=cos_deg)
#     _, sin_yi = lin_interp(features, 'lon', t=interp_t, mask=mask, func=sin_deg)
#     y3 = np.degrees(np.arctan2(sin_yi, cos_yi))
#     # delta times
#     xp = util.compute_xp(features, mask)
#     added_dts = util.delta_times(xi, xp) / 60.0
#     _, base_dts = lin_interp(features, 'min_dt_min', t=interp_t, mask=None)
#     y4 = (added_dts + base_dts) * 60 # in seconds for compatibility.
#     #
#     y = np.transpose([y0, y1, y2, y3, y4])
#     #
#     # Quick and dirty nearest neighbor (only works for binary labels I think)
#     if skip_label:
#         label_i = defined_i = None
#     else:
#         add_obj_data(obj, features)
#         _, raw_label_i = lin_interp(features, 'is_fishing', t=interp_t, 
#                                     mask=None) # is it a set
#         label_i = raw_label_i > 0.5

#         _, raw_defined_i = lin_interp(features, 'is_defined', t=interp_t, 
#                                       mask=None) # is it a set
#         defined_i = raw_defined_i > 0.5
#     #
#     return interp_t, xi, y, label_i, defined_i

def cook_features(raw_features, angle=None, mean_2 = None, mean_3 = None, noise=None, far_time=3 * 10 * minute):
    speed = raw_features[:, 0]
    angle = np.random.uniform(0, 2*np.pi) if (angle is None) else angle
    angle_feat = angle + (np.pi / 2.0 - raw_features[:, 1])
    
    if mean_2 is None:
        mean_2 = raw_features[:, 2].mean()
    if mean_3 is None:
        mean_3 = raw_features[:, 3].mean()
    d1 = raw_features[:, 2] - mean_2
    d2 = raw_features[:, 3] - mean_3
    dir_a = np.cos(angle) * d2 - np.sin(angle) * d1
    dir_b = np.cos(angle) * d1 + np.sin(angle) * d2

    if noise is None:
        noise = np.random.normal(0, .05, size=len(raw_features[:, 4]))
    noisy_time = np.maximum(raw_features[:, 4] / float(far_time) + noise, 0)
    is_far = np.exp(-noisy_time) 
    return np.transpose([speed, 
                      np.cos(angle_feat), 
                      np.sin(angle_feat),
                      dir_a,
                      dir_b,
                      is_far
                      ]), angle, mean_2, mean_3

def convert_from_features(features, obj=None):
    # Filter features down to just the ssvid / time span we want
    ssvid = os.path.basename(path).split('_')[0]
    mask = (features.ssvid == ssvid)
    features = features[mask]
    features = features.sort_values(by='timestamp')
    if obj is not None:
        timestamps = [x.to_pydatetime() for x in features.timestamp]
        t0 = obj['timestamp'].iloc[0].to_pydatetime()
        t1 = obj['timestamp'].iloc[-1].to_pydatetime()
        i0 = np.searchsorted(timestamps, t0, side='left')
        i1 = np.searchsorted(timestamps, t1, side='right')
        features = features.iloc[i0:i1]
        # Add fishing data to features
        add_obj_data(obj, features)
    # Rename so we can use featurs as obj:
    obj = pd.DataFrame({
        'timestamp' : features.timestamp,
        'speed' : features.speed_knots,
        'course' : features.course_degrees,
        'lat' : features.lat,
        'lon' : features.lon,
        'fishing' : features.fishing,
        })

def load_single_data(path, delta, skip_label=False, keep_frac=1, features=None):
    obj_tv = util.load_data(path, vessel_label=None)  
    obj = util.convert_from_legacy_format(obj_tv)
    obj['fishing'] = obj_tv['fishing']
    # if features is None:
    if features is not None:
        # Filter features down to just the ssvid / time span we want
        ssvid = os.path.basename(path).split('_')[0]
        mask = (features.ssvid == ssvid)
        features = features[mask]
        features = features.sort_values(by='timestamp')
        timestamps = [x.to_pydatetime() for x in features.timestamp]
        t0 = obj['timestamp'].iloc[0].to_pydatetime()
        t1 = obj['timestamp'].iloc[-1].to_pydatetime()
        i0 = np.searchsorted(timestamps, t0, side='left')
        i1 = np.searchsorted(timestamps, t1, side='right')
        features = features.iloc[i0:i1]
        # Add fishing data to features
        add_obj_data(obj, features)
        # Rename so we can use featurs as obj:
        obj = pd.DataFrame({
            'timestamp' : features.timestamp,
            'speed' : features.speed_knots,
            'course' : features.course_degrees,
            'lat' : features.lat,
            'lon' : features.lon,
            'fishing' : features.fishing,
            })

    t, x, y_tv, label, is_defined = build_features(obj, delta=delta, 
                                            skip_label=skip_label, keep_frac=keep_frac)
    # else:
    #     ssvid = os.path.basename(path).split('_')[0]
    #     mask = features.ssvid == ssvid
    #     if not mask.sum():
    #         print('skipping', ssvid)
    #         return None
    #     some_features = features[mask]
    #     some_features = some_features.sort_values(by='timestamp')
    #     timestamps = [x.to_pydatetime() for x in some_features.timestamp]
    #     t0 = obj['timestamp'].iloc[0].to_pydatetime()
    #     t1 = obj['timestamp'].iloc[-1].to_pydatetime()
    #     i0 = np.searchsorted(timestamps, t0, side='left')
    #     i1 = np.searchsorted(timestamps, t1, side='right')
    #     some_features = some_features.iloc[i0:i1]
    #     if len(some_features) < 2:
    #         print('skipping 2', ssvid)
    #         return None
    #     t, x, y_tv, label, is_defined = build_features_2(obj, some_features, 
    #                                             skip_label=skip_label, keep_frac=keep_frac)
    t = np.asarray(t)

    return (t, x, y_tv, label, is_defined)
    

def cook_single_data(t, x, y_tv, label, is_defined, start_ndx=0, end_ndx=None):
    t, x, y_tv, label, is_defined = [v[start_ndx:end_ndx] for v in 
                                                   (t, x, y_tv, label, is_defined)]     
    features_tv, angle, mean_2, mean_3 = cook_features(y_tv)
    return t, features_tv

    

def generate_data(sets, window, delta, min_samples, label_window=None, seed=888, skip_label=False,
                 keep_fracs=(1,), noise=None, force_local=False, precomp_features=None):
    #paths = make_paths(sets) #dropping for now to pass single path
    paths = sets
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
    min_ndx = 0
    for p in paths:
        for kf in keep_fracs:
            
            paired_data = load_single_data(p, delta, skip_label, kf, 
                                features=precomp_features)
            if paired_data is None:
                continue
            (t_tv, x_tv, y_tv, label_tv, defined_tv) = paired_data
            
            ### CHANGE could be len(x_tv) or len(x_fv).. they should be the same
            max_ndx = len(x_tv) - window_pts
            if max_ndx < min_ndx:
                print("skipping", p, "because it is too short")
                continue
            for _ in range(subsamples):
                KEEP_TRYS = 1000
                use_local = np.random.choice([True, False])
                for trys_left in reversed(range(KEEP_TRYS)):
                    ndx = np.random.randint(min_ndx, max_ndx + 1)                
                    if trys_left > 0:
                        if defined_tv[ndx:ndx+window_pts].sum() < window_pts // 2:
                            continue
                        if force_local:
                            if use_local == bool(label_tv[ndx:ndx+window_pts].sum()):
                                continue
                    t_chunk, f_chunk = cook_single_data(*paired_data, start_ndx=ndx, 
                                                                         end_ndx=ndx+window_pts)
                    times.append(t_chunk) 
                    features.append(f_chunk)

                    if skip_label:
                        transshipping.append(None)
                        labels.append(None)
                        defined.append(None)
                    else:
                        #######  CHANGES combine tv and fv
                        transshipping.append(label_tv[ndx:ndx+window_pts]) 
                        windowed_labels = label_tv[ndx+lbl_offset:ndx+lbl_offset+lbl_pts]

                        lbl = windowed_labels.mean() > 0.5
                        labels.append(lbl)

                        ###### CHANGES
                        windowed_defined = defined_tv[ndx+lbl_offset:ndx+lbl_offset+lbl_pts]
                        dfd = windowed_defined.mean() > 0.5
                        defined.append(dfd)
                        ######
                    break
                    
    return times, np.array(features), np.array(labels), np.array(transshipping), np.array(defined)  ### CHANGE
