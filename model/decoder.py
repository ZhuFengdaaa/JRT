import tensorflow as tf
import numpy as np
import logging


def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format) + b

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b

def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


class Decoder():
    def __init__(self, X, feature_size, device=None, thread_index=None, nlstm=128, layer_norm=False):
        self.sess = sess or tf.get_default_session()
        if X is not None:
            self.X = X
        else:
            self.X = tf.placeholder("float", [None, feature_size], name='X')
        self.dec_Z = tf.placeholder(tf.float32, [None, enc_space])
        # _h = tf.concat([self.X, self.dec_Z], 1)
        h = tf.layers.flatten(self.X)
        self.dec_M = tf.placeholder(tf.float32, [None]) #mask (done t-1)
        self.dec_S = tf.placeholder(tf.float32, [None, 2*nlstm]) # states
        z_prob = tf.fill([None, enc_space], 1.0/enc_space)
        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(self.dec_M, nenv, nsteps)
        if layer_norm:
            h5, self.snew = utils.lnlstm(xs, ms, self.dec_S, scope='lnlstm', nh=nlstm)
        else:
            h5, self.snew = utils.lstm(xs, ms, self.dec_S, scope='lstm', nh=nlstm)
            # h5, self.snew = utils.lstm(xs, ms, self.dec_S, scope='lstm', nh=nlstm)
        h = seq_to_batch(h5)
        self.h1 = fc(h, 'fc1', nh=enc_space, init_scale=np.sqrt(2))
        logging.info("self.h1.shape", self.h1.shape)
        logging.info("self.dec_Z", self.dec_Z)
        self.mp_h1 = tf.reduce_mean(self.h1)
        # logq = self.dec_Z * tf.math.log(tf.nn.softmax(self.h1))
        cls_loss = - tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.dec_Z, logits=self.mp_h1)
        self.cls_loss = tf.reduce_mean(cls_loss)
        # eheck shape
        self.initial_state = np.zeros(self.dec_S.shape.as_list(), dtype=float)
        # feed_dic = {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}
