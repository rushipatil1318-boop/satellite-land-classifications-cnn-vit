import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="LandProject")
class AddPositionEmbedding(layers.Layer):
    def build(self, input_shape):
        self.position_embedding = self.add_weight(name="position_embedding", shape=(1, input_shape[1], input_shape[2]), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return inputs + self.position_embedding


def make_cnn():
    inputs = layers.Input((64, 64, 3))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(name="cnn_features")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="keras_cnn")


def make_hybrid(cnn):
    backbone = tf.keras.Model(cnn.input, cnn.get_layer("cnn_features").output, name="frozen_cnn")
    backbone.trainable = False
    inputs = layers.Input((64, 64, 3))
    features = backbone(inputs)
    x = layers.Reshape((64, 64))(features)
    x = AddPositionEmbedding()(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + attention)
    mlp = layers.Dense(128, activation="gelu")(x)
    mlp = layers.Dense(64)(mlp)
    x = layers.LayerNormalization()(x + mlp)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="keras_cnn_vit")
