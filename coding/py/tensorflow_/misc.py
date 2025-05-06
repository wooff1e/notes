import tensorflow as tf



def build_dynamic_graph(model, shape=(None, None, 3)):
    x = tf.keras.layers.Input(shape=shape)
    model = tf.keras.models.Model(inputs=x, outputs=model(x))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model