from __future__ import division
from __future__ import print_function
import keras
from keras.models import Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, ELU, Input, Conv1D
from keras.layers.core import Activation
from keras import optimizers
from .util import hour
from .base_model import BaseModel, hybrid_pool_layer


class ConvNetModel5(BaseModel):
    delta = hour
    window = (29 + 4*6) *  delta
    time_points = window // delta - 1
    base_filter_count = 32
    fc_nodes = 512

    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(self.time_points, 6))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer(y)
        
        depth = 2 * depth 
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer(y)
        
        y = Flatten()(y)
        y = Dense(self.fc_nodes)(y)
        y = ELU()(y)
        y = keras.layers.BatchNormalization(scale=False, center=False)(y)

        y = Dense(self.fc_nodes)(y)
        y = ELU()(y)
        y = Dropout(0.5)(y)

        y = Dense(1)(y)
        y = Activation('sigmoid')(y)
        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        #opt = optimizers.Adam(lr=0.01, decay=0.5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  