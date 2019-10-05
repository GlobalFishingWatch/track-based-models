from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import h5py
import os
import shutil
import subprocess
import tempfile
import keras
from keras.layers import ELU, Conv1D, MaxPooling1D, AveragePooling1D
from keras import backend as K
from keras.models import load_model
from . import util
from .util import minute, hour, add_predictions

assert K.image_data_format() == 'channels_last'

EPSILON = 1e-10

class Normalizer(object):
    
    def fit(self, features):
        features = np.asarray(features)
        self.mean = features.mean(axis=(0, 1), keepdims=True)
        self.std = features.std(axis=(0, 1), keepdims=True)
        return self
        
    def norm(self, features):
        features = np.asarray(features)
        return (features - self.mean) / (self.std + EPSILON)
    
    def save(self, path, mode='w'):
        with h5py.File(path, mode) as f:
            if 'normalizer' not in f.keys():
                f.create_dataset('normalizer', [])
            f['normalizer'].attrs['mean'] = self.mean
            f['normalizer'].attrs['std'] = self.std
        
    @classmethod
    def load(cls, path):
        obj = cls()
        with h5py.File(path, 'r') as f:
            obj.mean = f['normalizer'].attrs['mean']
            obj.std = f['normalizer'].attrs['std']
        return obj


def hybrid_pool_layer(x, pool_size=2):
    return Conv1D(int(x.shape[-1]), 1)(
        keras.layers.concatenate([
            MaxPooling1D(pool_size, strides=2)(x),
            AveragePooling1D(pool_size, strides=2)(x)]))

def hybrid_pool_layer_2(x):
    depth = int(x.shape[-1])
    x2 = Conv1D(depth, 3, strides=2)(x)
    x2 = ELU()(x2)
    x2 = keras.layers.BatchNormalization(scale=False, center=False)(x2)
    return Conv1D(depth, 1)(keras.layers.concatenate([
                                      MaxPooling1D(3, strides=2)(x), x2]))    
    

class BaseModel(object):
    
    util = util

    def flatten(self, x):
        x = np.asarray(x)
        return x.reshape(x.shape[0], -1)
    
    def save(self, path):
        self.model.save(path)
        self.normalizer.save(path, 'r+')
        
    _mdl_cache = {}
    @classmethod
    def load(cls, path, refresh_cache=True):
        if path not in cls._mdl_cache:
            tempdir = tempfile.mkdtemp()
            try:
                if path.startswith('gs://'):
                    local_path = os.path.join(tempdir, os.path.basename(path))
                    subprocess.check_call(['gsutil', 'cp', path, local_path])
                else:
                    local_path = path
                mdl = cls()
                mdl.model = load_model(local_path)
                mdl.normalizer = Normalizer.load(local_path)
                cls._mdl_cache[path] = mdl
            finally:
                shutil.rmtree(tempdir)
        return cls._mdl_cache[path]


    def preprocess(self, x):
        x = np.asarray(x) 
        return x
    
    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0,
			verbose=1):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        return self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, sample_weight=sample_weight,
                      validation_split=validation_split, validation_data=validation_data, verbose=verbose)


    def predict(self, x):
        x1 = self.preprocess(x)
        return self.model.predict(x1)[:, :, 0]

    def _predict_from_features(self, features):
        base_shape = features[0].shape
        # Start from zero to ensure `i` always set
        for i in range(0, len(features)):
            if features[i].shape != base_shape:
                break
        features_list = [features[:i], features[i:]]
        chunks = []
        for features in features_list:
            if len(features) == 0:
                continue
            try:
                chunks.extend(self.predict(features))
            except:
                logging.debug('prediction failed: \n' +
                              'np.shape(features): {}\n'.format(np.shape(features))  
                              )
                raise
        return np.concatenate(chunks)

    def _predict_from_data(self, data, max_deltas=0):
        predictions = []
        for angle in [77, 167, 180, 270]:
            features, times = self.create_features_and_times(data, angle=angle,
                                        max_deltas=max_deltas)
            if len(features):
                predictions.append(self._predict_from_features(features))
        if len(predictions) == 0:
            predictions = np.array([], dtype=bool)
        else:
            predictions = np.mean(predictions, axis=0) > 0.5
        return times, predictions

    def predict_from_data(self, data, target_width=1):
        max_deltas = int(round((target_width - 1) / self.time_point_delta))
        return self._predict_from_data(data, max_deltas)

    def augment_data_with_predictions(self, data):
        """Add predictions to data

        Parameters
        ----------
        data : Pandas DataFrame having the following columns
               derived from AIS data:
                    timestamp : str
                    lat : float
                    lon : float
                    speed : float
                    course : float
                The data should be sorted by timestamp.

        """
        times, predictioms = self.predict_set_times(data)
        add_predictions(data, self.delta, times, predictions)

