import math
import tensorflow as tf
from torch.nn.modules.batchnorm import _BatchNorm
from tensorflow.keras.layers import BatchNormalization, Conv2D, Reshape, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal, Ones, Zeros
from tensorflow.keras import backend as K

class ConvBNReLU(tf.keras.Model):
    '''Module for the Conv-BN-ReLU tuple.'''
    def __init__(self, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = Conv2D(
            c_out, kernel_size=kernel_size, strides=stride,
            padding='same', dilation_rate=dilation, use_bias=False)
        self.bn = BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = K.relu(x)
        return x
    

class CrossEntropyLoss2d(tf.keras.Model):
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=reduction)
        self.ignore_index = ignore_index

    def call(self, inputs, targets):
        mask = tf.math.not_equal(targets, self.ignore_index)
        targets = tf.where(mask, targets, 0)
        loss = self.nll_loss(tf.nn.log_softmax(inputs), targets)
        return loss

class EMAU(tf.keras.Model):
    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        # Initialize your parameters here

        mu = K.random_normal_variable(shape=(1, c, k), mean=0, scale=math.sqrt(2. / k))
        mu = self._l2norm(mu, dim=1)
        self.mu = K.variable(mu, dtype='float32')

        self.conv1 = Conv2D(c, (1, 1), padding='same', activation='linear', use_bias=True)
        self.conv2 = Sequential([Conv2D(3, (1, 1), padding='same', activation='linear', use_bias=False),
                                BatchNormalization()])

        for m in self.layers:
            if isinstance(m, Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.filters
                m.kernel_initializer = RandomNormal(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNormalization):
                m.gamma_initializer = Ones()
                if m.beta_initializer is not None:
                    m.beta_initializer = Zeros()

    def call(self, x):
        idn = x

        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        _, c, h, w = K.int_shape(x)
        x = Reshape((c, h * w))(x)
        mu = self.mu

        for i in range(self.stage_num):
            x_t = K.permute_dimensions(x, (0, 2, 1))  # b * n * c
            z = K.batch_dot(x_t, mu, axes=[2, 1])  # b * n * k
            z = Softmax(axis=2)(z)  # b * n * k
            z_ = z / (1e-6 + K.sum(z, axis=1, keepdims=True))
            z_ = z / (1e-6 + K.sum(z, axis=2, keepdims=True))
            mu = K.batch_dot(x, z_, axes=[2, 1])  # b * c * k
            mu = self._l2norm(mu, dim=1)

        z_t = K.permute_dimensions(z, (0, 2, 1))  # b * k * n
        x = K.batch_dot(mu, z_t)  # b * c * n
        x = K.reshape(x, (-1, c, h, w))
        x = K.relu(x)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = K.relu(x)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + K.sqrt(K.sum(K.square(inp), axis=dim, keepdims=True)))
