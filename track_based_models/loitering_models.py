from __future__ import division
from __future__ import print_function
import datetime
import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model as KerasModel
from keras.layers import Dense, Dropout, Flatten, ELU, ReLU, Input, Conv1D
from keras.layers import BatchNormalization, MaxPooling1D, Concatenate
from keras.layers.core import Activation
from keras import optimizers
from .util import hour, minute
from .base_model import hybrid_pool_layer_2, Normalizer
from .single_track_model import SingleTrackModel
from . import util
from .util import minute, lin_interp, cos_deg, sin_deg 


from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K

from keras.utils.generic_utils import get_custom_objects


class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})


class LoiteringModelV1(SingleTrackModel):
    
    delta = 10 * minute
    window = 12 * hour + delta
    time_points = window // delta

    base_filter_count = 64
    fc_nodes = 512

    data_source_lbl='transshipping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(self.time_points, 6))
        y = input_layer
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        y = Flatten()(y)
        
        y = Dense(self.fc_nodes)(y)
        y = ELU()(y)

        y = Dense(1)(y)
        y = Activation('sigmoid')(y)
        output_layer = y
        model = KerasModel(inputs=input_layer, outputs=output_layer)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        #opt = optimizers.Adam(lr=0.01, decay=0.5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  





class LoiteringModelV2(SingleTrackModel):
    
    delta = 10 * minute
    window = 12 * hour + delta
    time_points = window // delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 7)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 5)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = hybrid_pool_layer_2(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 1)(y)
        y = ELU()(y)
        y = Conv1D(depth, 1)(y)
        y = ELU()(y)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        # 1 -> (3 -> 11) -> (23 -> 31) -> (63, 77) -> 73
        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), 1, 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
  


class LoiteringModelV3(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 93 # 72 = 12 hours, 120 = 20 hours, should be odd
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(self.time_points, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o2 = y
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o3 = y
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (4 -> 8) -> (16 -> 20) -> (40 -> 44) -> (88, 93) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 7.5 / 8
        y = keras.layers.Concatenate()([y, 
                            keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ELU()(y) # 3.5 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = keras.layers.UpSampling1D(2)(y) # 7 / 8
        y = keras.layers.Concatenate()([y, 
                            keras.layers.Cropping1D((16,16))(o2)])
        y = Conv1D(d2, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ELU()(y) # 3 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = keras.layers.UpSampling1D(2)(y) # 6 / 8
        y = keras.layers.Concatenate()([y, 
                            keras.layers.Cropping1D((40,40))(o1)])
        y = Conv1D(d1, 3)(y)
        y = ELU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ELU()(y) # 2 / 4


        y = Conv1D(1, 4)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.0005, schedule_decay=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape((len(labels), 1, 1))
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
  

import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x    


class LoiteringModelV4(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 93 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o2 = y
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o3 = y
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (4 -> 8) -> (16 -> 20) -> (40 -> 44) -> (88, 93) 

        # Upward branch 

        y = Conv1DTranspose(y, d3, 2, strides=2, padding='same') # 7.5 / 8
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y) # 3.5 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = Conv1DTranspose(y, d2, 2, strides=2, padding='same') # 7 / 8
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((16,16))(o2)])
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y) # 3 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = Conv1DTranspose(y, d1, 2, strides=2, padding='same') # 6 / 8
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((40,40))(o1)])
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y) # 2 / 4
        y = BatchNormalization(scale=False, center=False)(y)

        y = Conv1D(1, 4)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        print(np.shape(x1), np.shape(labels))
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self



class LoiteringModelV5(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 101 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o2 = y
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        o3 = y
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (5 -> 9) -> (18 -> 22) -> (44 -> 48) -> (96, 101) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y) # 3.5 / 4
        y = BatchNormalization(scale=False, center=False)(y) # 5

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((17,17))(o2)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y) # 3 / 4
        y = BatchNormalization(scale=False, center=False)(y) # 5

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((43,43))(o1)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y) # 2 / 4
        y = BatchNormalization(scale=False, center=False)(y) # 5

        y = Conv1D(1, 5)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self


class LoiteringModelV6(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 101 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        input_layer = Input(shape=(None, 6))

        d1 = depth = self.base_filter_count        
        y = input_layer
        y = Conv1D(depth, 4, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = o2 = Dropout(0.1)(y)
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = o3 = Dropout(0.2)(y)
        y = MaxPooling1D(2)(y)
        
        depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Dropout(0.3)(y)


        # (5 -> 9) -> (18 -> 22) -> (44 -> 48) -> (96, 101) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((17,17))(o2)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((43,43))(o1)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)

        y = Conv1D(1, 5)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05,
                               clipnorm=1.)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
    


class LoiteringModelV7(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 101 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        input_layer = Input(shape=(None, 6))

        d1 = depth = self.base_filter_count        
        y = input_layer
        y = Conv1D(depth, 4, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o1 = Dropout(0.1)(y)
        y = MaxPooling1D(2)(y)
        
        d2 = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o2 = Dropout(0.2)(y)
        y = MaxPooling1D(2)(y)
        
        d3 = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        o3 = Dropout(0.3)(y)
        y = MaxPooling1D(2)(y)

        depth = 3 * depth // 2
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Conv1D(depth, 3, activation='relu')(y)
        y = Dropout(0.4)(y)


        # (5 -> 9) -> (18 -> 22) -> (44 -> 48) -> (96, 101) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)
        y = Conv1D(d3, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((17,17))(o2)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)
        y = Conv1D(d2, 3, activation='relu')(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((43,43))(o1)])
        y = Conv1D(d3, 2, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)
        y = Conv1D(d1, 3, activation='relu')(y)

        y = Conv1D(1, 5)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.1,
                               clipnorm=0.1)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self


class LoiteringModelV8(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 65 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        
        y = Conv1D(depth, 3, dilation_rate=2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3, dilation_rate=2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        
        y = Conv1D(depth, 3, dilation_rate=4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3, dilation_rate=4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        
        y = Conv1D(depth, 3, dilation_rate=8)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3, dilation_rate=8)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        y = Conv1D(depth, 3, dilation_rate=1)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3, dilation_rate=1)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (1 - 5) (5 37) (37 53) (53 61) (61 65)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self


class LoiteringModelV9(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 65 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 64

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth, 3)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = MaxPooling1D(3, strides=1)(y)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)


        
        y = Conv1D(depth, 3, dilation_rate=2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth, 3, dilation_rate=2)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = MaxPooling1D(5, strides=1)(y)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)
        
        y = Conv1D(depth, 3, dilation_rate=4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth, 3, dilation_rate=4)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = MaxPooling1D(9, strides=1)(y)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)

        y = Conv1D(depth, 3, dilation_rate=8)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth, 3, dilation_rate=8)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = MaxPooling1D(17, strides=1)(y)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)

        y = Conv1D(depth, 3, dilation_rate=1)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3, dilation_rate=1)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (1 - 5) (5 37) (37 53) (53 61) (61 65)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self


class LoiteringModelV10(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 65 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 1
    window = time_points * delta

    base_filter_count = 32

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth // 2, 3)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = Conv1D(depth // 2, 1)(y)
        y2 = MaxPooling1D(3, strides=1)(y2)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)


        
        y = Conv1D(depth, 3, dilation_rate=2)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth // 2, 3, dilation_rate=2)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = Conv1D(depth // 2, 1)(y)
        y2 = MaxPooling1D(5, strides=1)(y2)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)
        
        y = Conv1D(depth, 3, dilation_rate=4)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth // 2, 3, dilation_rate=4)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = Conv1D(depth // 2, 1)(y)
        y2 = MaxPooling1D(9, strides=1)(y2)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)

        y = Conv1D(depth, 3, dilation_rate=8)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y1 = Conv1D(depth // 2, 3, dilation_rate=8)(y)
        y1 = ReLU()(y1)
        y1 = BatchNormalization(scale=False, center=False)(y1)
        y2 = Conv1D(depth // 2, 1)(y)
        y2 = MaxPooling1D(17, strides=1)(y2)
        y = Concatenate()([y1, y2])
        y = Conv1D(depth, 1)(y)

        y = Conv1D(depth, 3, dilation_rate=1)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)
        y = Conv1D(depth, 3, dilation_rate=1)(y)
        y = ReLU()(y)
        y = BatchNormalization(scale=False, center=False)(y)

        # (1 - 5) (5 37) (37 53) (53 61) (61 65)

        y = Conv1D(1, 1)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self




class LoiteringModelV11(SingleTrackModel):
    
    delta = 10 * minute
    time_points = 101 # 72 = 12 hours, 120 = 20 hours, should be odd
    time_point_delta = 8
    window = time_points * delta

    base_filter_count = 60

    data_source_lbl='transshiping' 
    data_target_lbl='is_target_encounter'
    data_undefined_vals = (0, 3)
    data_defined_vals = (1, 2)
    data_true_vals = (1,)
    data_false_vals = (2,)
    data_far_time = 3 * 10 * minute
    
    def __init__(self):
        
        self.normalizer = None
        
        d1 = depth = self.base_filter_count
        groups = g1 = depth // 16
        
        input_layer = Input(shape=(None, 6))
        y = input_layer
        y = Conv1D(depth, 4)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        o1 = y
        y = MaxPooling1D(2)(y)
        
        d2 = depth = 16 * (depth // 5)
        groups = g1 = depth // 16
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        o2 = Dropout(0.1)(y)
        y = MaxPooling1D(2)(y)
        
        d3 = depth = 16 * (depth // 5)
        groups = g1 = depth // 16
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        o3 = Dropout(0.2)(y)
        y = MaxPooling1D(2)(y)
        
        d4 = depth = 16 * (depth // 5)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        y = Conv1D(depth, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=groups, scale=False, center=False)(y)
        y = Dropout(0.3)(y)

        # (5 -> 9) -> (18 -> 22) -> (44 -> 48) -> (96, 101) 

        # Upward branch 

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y,  keras.layers.Cropping1D((4,4))(o3)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=g1, scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=g1, scale=False, center=False)(y)
        y = Conv1D(d3, 3)(y)
        y = ReLU()(y) # 3.5 / 4
        y = GroupNormalization(groups=g1, scale=False, center=False)(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((17,17))(o2)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=g2, scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=g2, scale=False, center=False)(y)
        y = Conv1D(d2, 3)(y)
        y = ReLU()(y) # 3 / 4
        y = GroupNormalization(groups=g2, scale=False, center=False)(y)

        y = keras.layers.UpSampling1D(2)(y) # 10
        y = keras.layers.Concatenate()([y, keras.layers.Cropping1D((43,43))(o1)])
        y = Conv1D(d3, 2)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=g3, scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y)
        y = GroupNormalization(groups=g3, scale=False, center=False)(y)
        y = Conv1D(d1, 3)(y)
        y = ReLU()(y) # 2 / 4

        y = Conv1D(1, 5)(y)
        y = Activation('sigmoid')(y)

        model = KerasModel(inputs=input_layer, outputs=y)
        opt = optimizers.Nadam(lr=0.002, schedule_decay=0.05)
        # opt = keras.optimizers.SGD(lr=0.00001, momentum=0.9, 
        #                                 decay=0.5, nesterov=True)

        model.compile(optimizer=opt, loss='binary_crossentropy', 
            metrics=["accuracy"], sample_weight_mode="temporal")
        self.model = model  

    def fit(self, x, labels, epochs=1, batch_size=32, sample_weight=None, validation_split=0, validation_data=0):
        self.normalizer = Normalizer().fit(x)
        x1 = self.preprocess(x)
        l1 = np.asarray(labels).reshape(len(labels), -1, 1)
        if validation_data not in (None, 0):
            a, b, c = validation_data
            validation_data = self.preprocess(a), b, c
        self.model.fit(x1, l1, epochs=epochs, batch_size=batch_size, 
                        sample_weight=sample_weight,
                      validation_split=validation_split, 
                      validation_data=validation_data)
        return self
    