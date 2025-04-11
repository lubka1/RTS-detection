import tensorflow as tf
from tensorflow.keras import layers, Model

class WeightedAverage(layers.Layer):
    '''
    Averaging results in a more compact representation and can prevent the model from learning redundant features by forcing it to focus on shared important aspects of both sources.
    Averaging the features results in a lower number of channels compared to concatenation, which could be more efficient if you're concerned about memory and computation.
    '''

    def __init__(self, n_output):
        super(WeightedAverage, self).__init__()
        self.W = tf.Variable(initial_value=tf.random.uniform(shape=[1,1,n_output], minval=0, maxval=1),
            trainable=True) # (1,1,n_inputs)

    def call(self, inputs):

        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = layers.Concatenate(axis=-1)(inputs) # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1) # (1,1,n_inputs)
        # weights sum up to one on last dim

        return tf.reduce_sum(weights*inputs, axis=-1) # (n_batch, n_feat)
    
    

# Spatial and channel attention (local + global)
class GridAttention(Model):
    '''
    Since we're working with limited computational resources (a CPU), 
    keeping the inter_channels small is essential for efficiency. 
    A smaller inter_channels value (e.g., 16, 32) will reduce the computational burden and memory consumption 
    but might limit the expressiveness of the attention mechanism. This might work well for simpler tasks.
    A larger inter_channels value (e.g., 64, 128, or higher) allows for more expressive power in the attention mechanism, which could improve model performance, especially in tasks that require complex feature interactions. 
    However, this will also increase the memory and computational cost.
    '''
    def __init__(self, inter_channels=32):
        super(GridAttention, self).__init__()
        self.inter_channels = inter_channels
        
        self.theta_x = layers.Conv2D(inter_channels, kernel_size=2, strides=2, padding='same')
        self.upsample = layers.UpSampling2D(size=(2, 2))
        self.phi_g = layers.Conv2D(inter_channels, kernel_size=1, strides=1, padding='same')
        self.add = layers.Add()
        self.relu = layers.Activation('relu')
        self.attention_conv = layers.Conv2D(1, kernel_size=1)
        self.sigmoid = layers.Activation('sigmoid')
        self.multiply = layers.Multiply()
    
    def call(self, features, gating):
        # Upsample features to match gating size
        theta_x = self.theta_x(features)
        theta_x = self.upsample(theta_x)
        
        # Apply phi_g
        phi_g = self.phi_g(gating)
        
        # Add and apply attention
        add = self.add([theta_x, phi_g])
        attention_map = self.relu(add)
        attention_map = self.attention_conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention to features
        attention_weighted = self.multiply([features, attention_map])
        
        return attention_weighted   

# Channel-wise attention (global)
class ChannelGate(layers.Layer):
    """
    The idea behind this type of attention is to learn which channels in the feature map are important for the task, 
    and adjust their weights accordingly.
    The ChannelGate layer learns to focus on important channels and suppress less important ones, which can help improve the model's performance by allowing it to focus on the most relevant features.
    Reduction ratio: Smaller values (e.g., 16) are often used in lightweight models or when computational efficiency is a priority.

Larger values (e.g., 32, 64) might be chosen when the model can afford to handle more complexity and requires more expressive attention.
    """
    def __init__(self, gate_channels, reduction_ratio=32):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio

        # Define the MLP layers
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(gate_channels // reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(gate_channels, activation='sigmoid')

    def call(self, x):
        # Apply global average pooling
        avg_pool = self.global_avg_pool(x)

        # Pass through MLP to compute the attention map
        channel_att_raw = self.dense1(avg_pool)
        channel_att_raw = self.dense2(channel_att_raw)

        # Reshape the output to apply scaling on the input feature map
        scale = tf.expand_dims(tf.expand_dims(channel_att_raw, 1), 1)  # Add spatial dimensions
        scale = tf.broadcast_to(scale, tf.shape(x))  # Match the input feature map's shape

        # Scale the input features by the attention weights
        return x * scale


class ResidualCorrectionNet(Model):
    """
    The idea is to refine the fused features after they have already been combined, ensuring that important information from both sources is preserved and enhanced.

    If each feature map has distinct but complementary information (e.g., one captures texture, the other captures semantic context), 
    then a ResidualCorrectionNet can help preserve both sources while allowing for interaction and refinement. 
    The RCN is relatively lightweight compared to larger fusion layers (like full attention or dense connections). 
    If computational efficiency is important (for example, when working with limited resources), this could serve as an efficient yet effective way to merge and refine features before passing them forward in the network.
    The residual connection ensures that important features from both sources aren’t lost in the process, and the output of the fusion layer can then serve as a better input for subsequent layers.
    """
    def __init__(self, in_channels, num_classes=1):
        super(ResidualCorrectionNet, self).__init__()
        
        # Define a small residual CNN with a few convolutional layers
        self.conv1 = layers.Conv2D(64, kernel_size=3, padding='same', input_shape=(None, None, in_channels))
        self.conv2 = layers.Conv2D(64, kernel_size=3, padding='same')
        self.conv3 = layers.Conv2D(num_classes, kernel_size=1)  # Output channel size matches num_classes

    def call(self, feature_map1, feature_map2):
        # CHECK THIS WITH THE PAPER
        # Concatenate feature maps from two networks
        x = tf.concat([feature_map1, feature_map2], axis=-1)  # Concatenate along the channel dimension
     #   residual = x  # Save the original input for the residual connection
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        correction_map = self.conv3(x)  # Output correction map
     #   correction_map += residual  # Add the residual to the correction map
        return correction_map