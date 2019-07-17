from __future__ import division
from __future__ import print_function
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
    obj['raw_timestamps'] = obj['timestamps']
    obj['timestamps'] = obj['timestamp'] = [dateutil.parser.parse(x) for x in obj['timestamps']]
    mask = np.ones_like(obj['timestamp'])
    for field in ['sogs', 'courses']:
        mask &= [(x is not None) for x in obj[field]]
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

def lin_interp(obj, key, delta=None, t=None, mask=None, func=None):
    if t is not None:
        assert delta is None, 'only one of `delta` or `t` may be specified'
        # convert timestamp to seconds
        t = np.array([int((x - obj['timestamp'].iloc[0]).total_seconds()) for x in t])
    if delta is None:
        assert t is not None, 'only one of `delta` or `t` may be specified'
    else:
        assert delta % 1 == 0, 'delta must be a whole number of seconds'
        
    fp = obj[key]

    timestamps = as_datetime_seq(obj['timestamp'])
    assert is_sorted(timestamps), 'data must be sorted'
    if mask is not None:
        fp = [x for (i, x) in enumerate(fp) if mask[i]]
        timestamps = [x for (i, x) in enumerate(timestamps) if mask[i]]
    if func is not None:
        fp = func(fp)
    ts0 = timestamps[0]
    xp = np.array([int((ts - ts0).total_seconds()) for ts in timestamps])
    a_smidgen = 0.1 # Added so that we capture the last point if it's on an even delta
    
    if t is None:
        t = np.arange(xp[0], xp[-1] + a_smidgen, delta)
    
    return t, np.interp(t, xp, fp)
  

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

# TODO: check if this is used
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
        print(t0,  t1, i0, i1, len(timestamps), 
            timestamps.min(), timestamps.max())
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

def features_to_data(features, ssvid=None, t0=None, t1=None):
    if ssvid is not None or t0 is not None or t1 is not None:
        mask = 1
        if ssvid is not None:
            mask &= (features.id == ssvid)
        if t0 is not None:
            mask &= (features.timestamp >= t0)
        if t1 is not None:
            mask &= (features.timestamp <= t1)
        features = features[mask]

    if features.dtypes['timestamp'] == np.dtype('<M8[ns]'):
        timestamps = features.timestamp
    else:
        timestamps = [dateutil.parser.parse(x) for x in data.timestamp] 

    return pd.DataFrame({
        'timestamp' : timestamps,
        'speed' : features.speed_knots,
        'course' : features.course_degrees,
        'lat' : features.lat,
        'lon' : features.lon,
        })