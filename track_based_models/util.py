from __future__ import division
from __future__ import print_function
from collections import namedtuple
import datetime
import dateutil.parser
import json
import numpy as np
import os
import pandas as pd

minute = 60
hour = 60 * minute

def make_paths(sets, all_paths):
    return [x for x in all_paths if path_to_set(x) in sets and is_valid_path(x)]

def path_to_set(x):
    return '_'.join(os.path.splitext(os.path.basename(x))[0].split('_')[:6])

def is_sorted(x):
    x = np.asarray(x)
    ge = (x[1:] >= x[:-1])
    return np.alltrue(ge)

def is_sorted_strict(x):
    x = np.asarray(x)
    ge = (x[1:] > x[:-1])
    return np.alltrue(ge)

# Don't include the observer data because some it is flakey and I don't
# want to worry about which is which.
def is_valid_path(x):
    return 'Observer' not in os.path.basename(x)


def load_json_data(path, vessel_label):
    with open(path) as f:
        obj = json.loads(f.read())
    if vessel_label is not None:
        obj = obj[vessel_label]
    obj['mmsi'] = str(obj['mmsi'])
    obj['raw_timestamps'] = obj['timestamps']
    obj['timestamps'] = obj['timestamp'] = [dateutil.parser.parse(x) for x in obj['timestamps']]
    mask = np.ones_like(obj['timestamp'], dtype=bool)
    for field in ['sogs', 'courses']:
#	print(mask.dtype,  np.array([(x is not None) for x in obj[field]]).dtype)
        mask &= np.array([(x is not None) for x in obj[field]], dtype=bool)
    for field in ['timestamp', 'lats', 'lons', 'sogs', 'courses']:
        obj[field] = [x for (i, x) in enumerate(obj[field]) if mask[i]]
    return obj
  
load_data = load_json_data

def _to_datetime(x):
    if isinstance(x, pd.Series):
        return [x.to_pydatetime(x) for x in obj['timestamp']]
    return x

def as_datetime_seq(value):
    if isinstance(value, pd.Series):
        return [x.to_pydatetime() for x in value]
    else:
        return value

InterpInfo = namedtuple('InterpInfo',
    ['interp_seconds', 'raw_seconds', 
     'interp_timestamps', 'raw_timestamps', 'mask'])

def setup_lin_interp(obj, delta=None, timestamps=None, mask=None, offset_seconds=0):
    if delta is None:
        assert timestamps is not None, 'only one of `delta` or `t` may be specified'
    else:
        assert delta % 1 == 0, 'delta must be a whole number of seconds'

    raw_timestamps = as_datetime_seq(obj['timestamp'])
    assert is_sorted(raw_timestamps), 'data must be sorted'
    if mask is not None:
        raw_timestamps = [x for (i, x) in enumerate(raw_timestamps) if mask[i]]
    ts0 = raw_timestamps[0]
    raw_seconds = np.array([int((ts - ts0).total_seconds()) for ts in raw_timestamps])

    if timestamps is not None:
        assert delta is None, 'only one of `delta` or `timestamps` may be specified'
        # convert timestamp to seconds
        interp_seconds = np.array([int((x - obj['timestamp'].iloc[0]).total_seconds()) 
                                        for x in timestamps])
    else:
        a_smidgen = 0.1 # Added so that we capture the last point if it's on an even delta
        interp_seconds = np.arange(raw_seconds[0] + offset_seconds, raw_seconds[-1] + a_smidgen, delta)
    interp_timestamps = [raw_timestamps[0] + datetime.timedelta(seconds=int(x)) 
                            for x in interp_seconds]


    return InterpInfo(interp_seconds=interp_seconds, raw_seconds=raw_seconds,  
                      raw_timestamps=np.array(raw_timestamps), 
                      interp_timestamps=np.array(interp_timestamps),
                      mask=mask)

def do_lin_interp(obj, info, key, func=None):
    raw_data = obj[key]
    if info.mask is not None:
        raw_data = [x for (i, x) in enumerate(raw_data) if info.mask[i]]
    if func is not None:
        raw_data = func(raw_data)
    return np.interp(info.interp_seconds, info.raw_seconds, raw_data)

def do_degree_interp(obj, info, key):
    cos_yi = do_lin_interp(obj, info, key, func=cos_deg)
    sin_yi = do_lin_interp(obj, info, key, func=sin_deg)
    return np.degrees(np.arctan2(sin_yi, cos_yi))

def lin_interp(obj, key, delta=None, t=None, mask=None, func=None):
    if t is not None:
        assert delta is None, 'only one of `delta` or `t` may be specified'
        # convert timestamp to seconds
        t = np.array([int((x - obj['timestamp'].iloc[0]).total_seconds()) for x in t])
    if delta is None:
        assert t is not None, 'only one of `delta` or `t` may be specified'
    else:
        assert delta % 1 == 0, 'delta must be a whole number of seconds'
        

    timestamps = as_datetime_seq(obj['timestamp'])
    assert is_sorted(timestamps), 'data must be sorted'
    if mask is not None:
        timestamps = [x for (i, x) in enumerate(timestamps) if mask[i]]
    ts0 = timestamps[0]
    xp = np.array([int((ts - ts0).total_seconds()) for ts in timestamps])
    a_smidgen = 0.1 # Added so that we capture the last point if it's on an even delta
    if t is None:
        t = np.arange(xp[0], xp[-1] + a_smidgen, delta)

    fp = obj[key]
    if mask is not None:
        fp = [x for (i, x) in enumerate(fp) if mask[i]]
    if func is not None:
        fp = func(fp)

    return t, np.interp(t, xp, fp)
  

def interp_degrees(obj, key, delta=None, t=None, mask=None):
    xi, cos_yi = lin_interp(obj, key, delta=delta, mask=mask, func=cos_deg)
    _,  sin_yi = lin_interp(obj, key, delta=delta, mask=mask, func=sin_deg)
    return xi, np.degrees(np.arctan2(sin_yi, cos_yi))

def compute_xp(obj, mask):
    timestamps = as_datetime_seq(obj['timestamp'])
    if mask is not None:
        timestamps = [x for (i, x) in enumerate(timestamps) if mask[i]]
    ts0 = timestamps[0]
    xp = np.array([int((ts - ts0).total_seconds()) for ts in timestamps])
    return xp

def delta_times(ti, t):
    """Compute maximimum gap between ti (interpolated) and t (original)"""
    prv = np.clip(np.searchsorted(t, ti, side='left'), 0, len(t) - 1)
    nxt = np.clip(np.searchsorted(t, ti, side='right'), 0, len(t) - 1)
    dtp = abs(ti - t[prv])
    dtn = abs(ti - t[nxt])
    return np.minimum(dtn, dtp)

def cos_deg(x):
    return np.cos(np.radians(x))

def sin_deg(x):
    return np.sin(np.radians(x))

def add_predictions(data, times, predictions, delta, column):
    """Add predicted values to a copy of a dataframe"""
    preds = np.empty(len(data))
    preds.fill(np.nan)
    timestamps = [x.to_pydatetime() for x in data.timestamp]
    for t, p in zip(times, predictions):
        t0 = t - datetime.timedelta(seconds=delta // 2)
        t1 = t + datetime.timedelta(seconds=delta // 2)
        i0 = np.searchsorted(timestamps, t0, side='left')
        i1 = np.searchsorted(timestamps, t1, side='right')
        preds[i0:i1] = p
    data = data.copy()
    data[column] = preds
    return data


def convert_to_legacy_format(data):
    """Convert data frame in pipeline format to tool json"""
    mask = ~data.speed.isnull() & ~data.course.isnull()
    data = data[mask]
    return {
            'timestamps' : [x.to_pydatetime() for x in data.timestamp],
            'lats' : data.lat.values,
            'lons' : data.lon.values,
            'sogs' : data.speed.values,
            'courses' : data.course.values
        }


def convert_from_legacy_format(data):
    """Convert data frame in pipeline format to tool json"""
    mask = ~np.isnan(data['sogs']) & ~np.isnan(data['courses'])
    return pd.DataFrame({
            'timestamp' : [x for (i, x) in 
                    enumerate(data['timestamps']) if mask[i]],
            'lat' : np.asarray(data['lats'])[mask],
            'lon' : np.asarray(data['lons'])[mask],
            'speed' : np.asarray(data['sogs'])[mask],
            'course' : np.asarray(data['courses'])[mask]
        })

def add_predictions(data, delta, times, predictions, label='inferred'):
    preds = np.empty(len(data))
    preds.fill(np.nan)
    timestamps = [x.to_pydatetime() for x in data.timestamp]
    for t, p in zip(times, predictions):
        t0 = t - datetime.timedelta(seconds=delta // 2)
        t1 = t + datetime.timedelta(seconds=delta // 2)
        i0 = np.searchsorted(timestamps, t0, side='left')
        i1 = np.searchsorted(timestamps, t1, side='right')
        preds[i0:i1] = p
    data[label] = preds


def features_to_data(features, t0=None, t1=None):
    if t0 is not None or t1 is not None:
        mask = 1
        if t0 is not None:
            mask &= (features.timestamp >= t0)
        if t1 is not None:
            mask &= (features.timestamp <= t1)
        features = features[mask]
    else:
        features = features.copy()

    if features.dtypes['timestamp'] != np.dtype('<M8[ns]'):
        features['timestamp'] = [dateutil.parser.parse(x) for x in data.timestamp] 

    return features


def load_training_data(path, vessel_label, data_source_lbl):     
    raw_obj = load_json_data(path, vessel_label=vessel_label)  
    obj = convert_from_legacy_format(raw_obj)
    mask = (~np.isnan(np.array(raw_obj['sogs'])) & 
            ~np.isnan(np.array(raw_obj['courses'])))
    if data_source_lbl is not None:
        obj[data_source_lbl] = np.asarray(raw_obj[data_source_lbl])[mask]
    ssvid = raw_obj['mmsi']
    return ssvid, obj


