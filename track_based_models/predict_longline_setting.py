import datetime
import dateutil
import numpy as _np
import pandas as _pd
from .longline_data import build_features as _build_features
from .longline_data import cook_features as _cook_features
from .longline_sets_models import ConvNetModel5 as SetsModel
from .util import add_predictions as _add_predictions
from .util import features_to_data



_mdl_cache = {}
def predict_set_times(mdl, data):
    if isinstance(mdl, str):
        if mdl not in _mdl_cache:
            _mdl_cache[mdl] = SetsModel.load(mdl)
        mdl = _mdl_cache[mdl]
    predictions = []
    for angle in [77, 167, 180, 270]:
        features, times = mdl.create_features_and_times(data, angle=angle)
        predictions.append(mdl.predict(features))
    return times, _np.mean(predictions, axis=0) > 0.5


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

