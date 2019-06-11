from __future__ import division
from __future__ import print_function
import datetime
import dateutil.parser
from glob import glob
import json
import numpy as np
import os

from . import util
from .util import minute, lin_interp, cos_deg, sin_deg

all_paths = glob('./labeled_transshipment_tracks/*.json')

data_sets = sorted(set([util.path_to_set(x) for x in all_paths]))

np.random.seed(888)    
np.random.shuffle(data_sets)

n_sets = len(data_sets)
  

def build_features(obj, delta=None, interp_t=None, skip_label=False, keep_frac=1.0):
    n_pts = len(obj['lat'])
    assert 0 < keep_frac <= 1
    if keep_frac == 1:
        mask = None
    else:           
        # Build a random mask with probability keep_frac. Force
        # first and last point to be true so the time frame
        # stays the same.
        mask = np.random.uniform(0, 1, size=[n_pts]) < keep_frac
        mask[0] = mask[-1] = True 
        
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
        t = [(t0 + datetime.timedelta(seconds=delta * i)) for i in range(len(y1))]
    else:
        t = interp_t
    y = np.transpose([y0, y1, y2, y3, y4])
    #
    # Quick and dirty nearest neighbor (only works for binary labels I think)
    if skip_label:
        label_i = defined_i = None
    else:
        obj['is_defined'] = [0 if i in (0, 3) else 1 for i in obj['transshiping']]
        obj['is_target_encounter'] = [1 if i == 1 else 0 for i in obj['transshiping']]
        _, raw_label_i = lin_interp(obj, 'is_target_encounter', delta=delta, t=interp_t, mask=None, # Don't mask labels - use undropped labels for training 
                                    func=lambda x: np.array(x) == 1) # is it a set
        label_i = raw_label_i > 0.5

        _, raw_defined_i = lin_interp(obj, 'is_defined', delta=delta, t=interp_t, mask=None, # Don't mask labels - use undropped labels for training 
                                    func=lambda x: np.array(x) == 1) # is it a set
        defined_i = raw_defined_i > 0.5
    #
    return t, xi, y, label_i, defined_i
  
  

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


def load_single_data(path, delta, skip_label=False, keep_frac=1):
    obj = util.load_data(path, vessel_label= 'position_data_reefer')  
    data = util.convert_from_legacy_format(obj)
    data['transshiping'] = obj['transshiping']
    t, x, y, label, is_defined = build_features(data, delta=delta, 
                                            skip_label=skip_label, keep_frac=keep_frac)
        
    t = np.asarray(t)

    return (t, x, y, label, is_defined)
    

def cook_single_data(t, x, y_tv, label, is_defined, start_ndx=0, end_ndx=None):
    t, x, y_tv, label, is_defined = [v[start_ndx:end_ndx] for v in 
                                                   (t, x, y_tv, label, is_defined)]     
    features_tv, angle, mean_2, mean_3 = cook_features(y_tv)
    return t, features_tv

    

def generate_data(sets, window, delta, min_samples, label_window=None, seed=888, skip_label=False,
                 keep_fracs=(1,), noise=None, force_local=False):
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
            
            paired_data = load_single_data(p, delta, skip_label, kf)
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
                    if force_local and trys_left > 0:
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
