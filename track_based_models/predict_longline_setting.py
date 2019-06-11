import datetime
import dateutil
import numpy as _np
import pandas as _pd
from .longline_data import build_features as _build_features
from .longline_data import cook_features as _cook_features
from .longline_sets_models import ConvNetModel5 as SetsModel


def _create_features_and_times(mdl, data):
    t, xi, y, label_i, defined_i = _build_features(data, delta=mdl.delta, skip_label=True)
    min_ndx = 0
    max_ndx = len(t) - mdl.time_points
    features = []
    for i in range(min_ndx, max_ndx):
        raw_features = y[i:i+(mdl.time_points+1)]
        features.append(_cook_features(raw_features, angle=77, noise=0)[0])
    times = t[mdl.time_points//2:-mdl.time_points//2]
    return features, times


# TODO: use util.add_predictions
def _add_predictions(mdl, data, times, predictions):
    preds = _np.empty(len(data))
    preds.fill(_np.nan)
    timestamps = [x.to_pydatetime() for x in data.timestamp]
    for t, p in zip(times, predictions):
        t0 = t - datetime.timedelta(seconds=mdl.delta // 2)
        t1 = t + datetime.timedelta(seconds=mdl.delta // 2)
        i0 = _np.searchsorted(timestamps, t0, side='left')
        i1 = _np.searchsorted(timestamps, t1, side='right')
        preds[i0:i1] = p
    data = data.copy()
    data['inferred_setting'] = preds
    return data

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

    if features.dtypes['timestamp'] == _np.dtype('<M8[ns]'):
        timestamps = features.timestamp
    else:
        timestamps = [dateutil.parser.parse(x) for x in data.timestamp] 

    return _pd.DataFrame({
        'timestamp' : timestamps,
        'speed' : features.speed_knots,
        'course' : features.course_degrees,
        'lat' : features.lat,
        'lon' : features.lon,
        })


_mdl_cache = {}
def predict_set_times(mdl, data):
    if isinstance(mdl, str):
        if mdl not in _mdl_cache:
            _mdl_cache[mdl] = SetsModel.load(mdl)
        mdl = _mdl_cache[mdl]
    features, times = _create_features_and_times(mdl, data)
    predictions = mdl.predict(features)
    return times, predictions


def predict_ll_setting(mdl, data):
    """Predict if a longline is setting at different timepoints.

    Parameters
    ----------
    mdl : KerasModel | str
    data : Pandas DataFrame having the following columns
           derived from AIS data:
                timestamp : str
                lat : float
                lon : float
                speed : float
                course : float
            The data should be sorted by timestamp.
    """
    times, predictioms = predict_set_times(mdl, data)
    return _add_predictions(mdl, data, times, predictions)

