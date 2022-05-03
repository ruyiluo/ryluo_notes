import random
import numpy as np
import pandas as pd 
import networkx as nx 
from tqdm import tqdm
from gensim.models import Word2Vec

from alias import create_alias_tables, alias_sample
from feature_column import SparseFeat

import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal, Zeros, glorot_normal
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.layers import Flatten, Concatenate, Dense, Reshape

from features import FeatureEncoder
from tensorflow.python.keras import backend as K



class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        embeddings, inputs, label_idx = inputs_with_label_idx
        
        loss = tf.nn.sampled_softmax_loss(weights=embeddings,  # self.item_embedding.
                                          biases=self.zero_bias,
                                          labels=label_idx,
                                          inputs=inputs,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.size,  # self.target_song_size
                                          )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'num_sampled': self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)



class EGESPooling(Layer):
    def __init__(self, item_nums, feat_nums, l2_reg=0.001, seed=1024, **kwargs):
        super(EGESPooling, self).__init__(**kwargs)
        self.item_nums = item_nums
        self.feat_nums = feat_nums
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('`EGESPooling` layer should be called \
                on a list of at least 2 inputs')

        self.alpha_embeddings = self.add_weight(
              name='alpha_attention',
              shape=(self.item_nums, self.feat_nums),
              dtype=tf.float32, 
              initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=self.seed),
              regularizer=l2(self.l2_reg))

    def call(self, inputs, **kwargs):
        stack_embedding = Concatenate(axis=1)(inputs[0])  # (B, num_feate, embedding_size)
        item_input = inputs[1]  # (B * 1)
        alpha_embedding = tf.nn.embedding_lookup(self.alpha_embeddings, item_input) #(B * 1 * feat_nums)
        alpha_embedding = tf.transpose(alpha_embedding, perm=[0,2,1]) # (B,feat_nums,1)
        side_info_weight = tf.nn.softmax(alpha_embedding, axis=1) # exp, (B, feat_nums, 1)
        merge_embedding_matrix = tf.multiply(side_info_weight, stack_embedding) #(B, feat_nums, 1) * (B, feat_nums, embedding_size)
        merge_embedding = tf.reduce_sum(merge_embedding_matrix, axis=1, keepdims=False) # B, embedding_size
        return merge_embedding

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"item_nums": self.item_nums, "seed": self.seed, 
                  "feat_nums": self.feat_nums, "l2_reg": self.l2_reg}
        base_config = super(EGESPooling, self).get_config()
        base_config.update(config)
        return base_config



class EmbeddingIndex(Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class PoolingLayer(Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):

        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = concat_func(expand_seq_value_len_list)
        if self.mode == "mean":
            hist = reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None

