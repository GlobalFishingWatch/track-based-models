import datetime
import dateutil.parser
import numpy as _np
from .loitering_data import build_features as _build_features
from .loitering_data import cook_features as _cook_features
from .loitering_models import LoiteringModel
from .util import add_predictions as _add_predictions
from . import predict_1_track



def _create_features_and_times(mdl, data):
    return predict_1_track.create_features_and_times(mdl, data, 
                                        _build_features, _cook_features)


_mdl_cache = {}
def _predict_loitering(mdl, data):
    if isinstance(mdl, str):
        if mdl not in _mdl_cache:
            _mdl_cache[mdl] = LoiteringModel.load(mdl)
        mdl = _mdl_cache[mdl]
    if data.dtypes['timestamp'] != _np.dtype('<M8[ns]'):
        data = data.copy()
        data['timestamp'] = [dateutil.parser.parse(x) for x in data.timestamp] 
    features, times = _create_features_and_times(mdl, data)
    predictions = mdl.predict(features)
    return _add_predictions(data, times, predictions, mdl.delta, 'is_loitering'), times, predictions


def predict_loitering(mdl, data):
    """Predict if a transhipment vessel is loitering.

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
    return _predict_loitering(mdl, data)[0]
