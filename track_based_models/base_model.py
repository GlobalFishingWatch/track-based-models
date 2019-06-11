from __future__ import division
from __future__ import print_function
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
from .util import minute, hour

assert K.image_data_format() == 'channels_last'


class Normalizer(object):
    
    def fit(self, features):
        features = np.asarray(features)
        self.mean = features.mean(axis=(0, 1), keepdims=True)
        self.std = features.std(axis=(0, 1), keepdims=True)
        return self
        
    def norm(self, features):
        features = np.asarray(features)
        return (features - self.mean) / self.std
    
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
    
    def flatten(self, x):
        x = np.asarray(x)
        return x.reshape(x.shape[0], -1)
    
    def save(self, path):
        self.model.save(path)
        self.normalizer.save(path, 'r+')
        
    @classmethod
    def load(cls, path):
        tempdir = tempfile.mkdtemp()
        try:
            if path.startswith('gs://'):
                new_path = os.path.join(tempdir, os.path.basename(path))
                subprocess.check_call(['gsutil', 'cp', path, tempdir])
                path = new_path
            mdl = cls()
            mdl.model = load_model(path)
            mdl.normalizer = Normalizer.load(path)
            return mdl
        finally:
            shutil.rmtree(tempdir)

    def preprocess(self, x):
        x = np.asarray(x) # 3 / 4
        dxy = x[:, 1:, 3:5] - x[:, :-1, 3:5]
        x = 0.5 * (x[:, 1:, :] + x[:, :-1, :])
        x[:, :, 3:5] = dxy
        return x
    
    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, sample_weight=sample_weight,
                      validation_split=validation_split, validation_data=validation_data)
        return self

    def predict(self, x):
        x1 = self.preprocess(x)
        return self.model.predict(x1)[:, 0] > 0.5
