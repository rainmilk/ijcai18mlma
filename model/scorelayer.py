from keras.layers import Layer
import keras.backend as K


class ScoreLayer(Layer):
    def __init__(self, keepdims=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None, **kwargs):
        self.supports_masking = True
        self.keepdims = keepdims
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        super(ScoreLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[0][-1],),
                                      name='weight',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      constraint=self.kernel_constraint)
        super(ScoreLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1, 1) if self.keepdims else (input_shape[0][0], 1)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        e = inputs[0] * (inputs[1] + self.W)
        return K.sum(e, axis=-1, keepdims=self.keepdims)