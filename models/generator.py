import tensorflow as tf
from tensorflow.keras import layers

def conv_block(x, filters):
    # Convolutional block for U-Net
    x = layers.Conv3D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv3D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def build_unet_generator(input_shape, filters=32):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = conv_block(inputs, filters)
    pool1 = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = conv_block(pool1, filters*2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = conv_block(pool2, filters*4)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4 = conv_block(pool3, filters*8)
    pool4 = layers.MaxPooling3D(pool_size=(1, 2, 2))(conv4)
    
    # Bottleneck
    conv5 = conv_block(pool4, filters*16)
    
    # Decoder
    up6 = layers.Conv3DTranspose(filters*8, (1, 2, 2), strides=(1, 2, 2), padding='same')(conv5)
    up6 = layers.Concatenate()([up6, conv4])
    conv6 = conv_block(up6, filters*8)
    
    up7 = layers.Conv3DTranspose(filters*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
    up7 = layers.Concatenate()([up7, conv3])
    conv7 = conv_block(up7, filters*4)
    
    up8 = layers.Conv3DTranspose(filters*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
    up8 = layers.Concatenate()([up8, conv2])
    conv8 = conv_block(up8, filters*2)
    
    up9 = layers.Conv3DTranspose(filters, (1, 2, 2), strides=(1, 2, 2), padding='same')(conv8)
    up9 = layers.Concatenate()([up9, conv1])
    conv9 = conv_block(up9, filters)
    
    outputs = layers.Conv3D(1, 1, activation='tanh')(conv9)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)