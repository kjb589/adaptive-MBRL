import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *


# Is it ready?
# yes


# Building Blocks
class ResNetBlock3D(Layer):
    """
    Residual Blocks for Height, Width, and Time
    - Used to extract spatiotemporal features for state representation
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = Conv3D(out_channels, kernel_size=3, strides=stride, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv3D(out_channels, kernel_size=3, strides=1, padding='same')

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = Sequential([
                Conv3D(out_channels, kernel_size=1, strides=stride, padding='same'),
                BatchNormalization()
            ])


    def call(self, x, training=False):
        identity = x

        y = self.conv1(x)
        y = self.bn(y, training=training)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn(y, training=training)

        if self.downsample is not None:
            identity = self.downsample(identity)

        y = y + identity
        y = self.relu(y)

        return y


class TemporalEncoder(Layer):
    """
    Temporal Encoder
    - Positional Embedding equivalent for sequences for Attention mechanism
    """
    def __init__(self, embed_dim=128, max_len=100):
        super().__init__()
        self.max_len = max_len
        self.pos_embed = None
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(self.max_len, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x):
        T = tf.shape(x)[1]
        T = tf.minimum(T, self.max_len)
        return x[:, :T, :] + self.pos_embed[:T]


class MHA(Layer):
    """
    Multi-Head Attention
    - highlights relevant features for more accurate prediction
    """
    def __init__(self, num_heads, key_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def call(self, x, training=False):
        x = tf.transpose(x, [1, 0, 2])
        attn_out = self.attn(x, x, x, training=training)
        x = x + self.dropout(attn_out, training=training)
        return tf.transpose(self.norm(x), [1, 0, 2])


class MLP(Layer):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        # x: (T, B, D) — average over time
        x = tf.reduce_mean(x, axis=0)  # (B, D) — average over time
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)  # (B, 1), values ∈ [0, 1]

        return x


def spatial_pooling(x):
    shape = tf.shape(x)
    B, T, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
    x = tf.reshape(x, [-1,H,W,C])
    x = GlobalAveragePooling2D()(x)
    x = tf.reshape(x, [B,T,-1])
    return x


class AttentionMechanism(Model):
    """
    Attention Mechanism: Attention-based Gating Mechanism for Multi-Context MBRL framework
    - Uses 3D ResNet + Attention to solve for weights for gating dynamics models
    """
    def __init__(self, in_channels=3, embed_dim=128, num_heads=4, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.conv1 = Sequential([
            Conv3D(64, 3, strides=2, padding='same'),
            ReLU(),
            MaxPooling3D(3,2)
        ])

        self.resnet = Sequential([
            ResNetBlock3D(in_channels, 32, 1),
            ResNetBlock3D(32, 64, 2),
            ResNetBlock3D(64, 128, 2),
            Lambda(spatial_pooling),
        ])

        self.attention_mechanism = Sequential([
            TemporalEncoder(max_len=100),
            MHA(num_heads=num_heads, key_dim=embed_dim),
            MLP()
        ])


    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.resnet(x)
        latent = x
        weight = self.attention_mechanism(x)

        return {"latent": latent, "weight": weight} # Output shape [batch, weights]


    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config
