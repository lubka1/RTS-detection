import tensorflow as tf
from tensorflow.keras import layers, Model

class WeightedAverage(layers.Layer):
    """
    Weighted average fusion of multiple input tensors. 
    Averages the features from different sources using a learnable softmax weight.
    
    Args:
        n_output (int): Number of output channels.
    """
    def __init__(self, n_output):
        super(WeightedAverage, self).__init__()
        self.W = tf.Variable(
            initial_value=tf.random.uniform(shape=[1, 1, n_output], minval=0, maxval=1),
            trainable=True
        )

    def call(self, inputs):
        inputs = [tf.expand_dims(i, -1) for i in inputs]    # inputs: list of tensors of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        inputs = layers.Concatenate(axis=-1)(inputs)        # Shape: (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1)            # Shape: (1, 1, n_inputs)
        return tf.reduce_sum(weights * inputs, axis=-1)



class ChannelGate(layers.Layer):
    """
    Learns channel-wise attention weights for a feature map.

    Args:
        gate_channels (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for intermediate layer. Smaller values (e.g., 16) for light models.

    Channel attention mechanism adapted from the CBAM paper:
    https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
    """
    def __init__(self, gate_channels, reduction_ratio=32):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(gate_channels // reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(gate_channels, activation='sigmoid')

    def call(self, x):
        avg_pool = self.global_avg_pool(x)
        channel_att_raw = self.dense1(avg_pool)
        channel_att_raw = self.dense2(channel_att_raw)
        scale = tf.expand_dims(tf.expand_dims(channel_att_raw, 1), 1)
        scale = tf.broadcast_to(scale, tf.shape(x))
        return x * scale


class ResidualCorrectionNet(Model):
    """
    Refines fused features using a residual structure of convolutions.

    Args:
        in_channels (int): Number of input channels (from concatenated feature maps).
        num_classes (int): Number of output channels/classes.
    
    Adapted from Audebert et al.:https://arxiv.org/pdf/1609.06846 , https://arxiv.org/pdf/1711.08681v1 .
    """
    def __init__(self, in_channels, num_classes=1):
        super(ResidualCorrectionNet, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(None, None, in_channels))
        self.conv2 = layers.Conv2D(64, kernel_size=3, padding='same')
        self.conv3 = layers.Conv2D(num_classes, kernel_size=1)

    def call(self, feature_map1, feature_map2):
        x = tf.concat([feature_map1, feature_map2], axis=-1)
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        correction_map = self.conv3(x)
        return correction_map
