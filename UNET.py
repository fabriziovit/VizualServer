import keras.initializers.initializers_v1
import tensorflow as tf
import numpy as np
import pickle as pkl
from skimage.io import imread
from os import listdir
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, Conv2DTranspose, \
    Dropout
#from keras import initializers
#from keras import optimizer_v2
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras import backend




@tf.custom_gradient
def MyLoss(y_true, y_pred):


    #return tf.keras.losses.MSLE(y_true, y_pred)
    #return tf.sqrt(tf.keras.losses.MSE(y_true, y_pred))
    #return tf.keras.losses.MAE(y_true, y_pred)
    #return tf.keras.losses.MSE(y_true, y_pred)
    #return 10./tf.image.psnr(y_true, y_pred, max_val=1.0)
    #return 100/tf.image.psnr(y_true, y_pred, max_val=1.0) * (1. - (tf.image.ssim(y_true, y_pred, max_val=1.0) + 1.)/2.)
    #return (1. - (tf.image.ssim(y_true, y_pred, max_val=1.0) + 1.)/2.)
    #ss = (1. - (tf.image.ssim(y_true, y_pred, max_val=1.0) + 1.)/2.)
    c1t, c2t = tf.image.image_gradients(y_true)
    ct = tf.math.sqrt(tf.math.add(tf.math.square(c1t), tf.math.square(c2t)))
    #ct = tf.math.add(tf.math.abs(c1t), tf.math.abs(c2t))
    c1y, c2y = tf.image.image_gradients(y_pred)
    #cy = tf.math.add(tf.math.abs(c1y), tf.math.abs(c2y))
    cy = tf.math.sqrt(tf.math.add(tf.math.square(c1y), tf.math.square(c2y)))
    df = tf.math.add(tf.math.subtract(y_true, y_pred), tf.math.subtract(ct, cy))
    @tf.function
    def grad(upstream):
        dldy =  tf.math.divide(tf.math.multiply(tf.math.sign(tf.math.subtract(y_pred, y_true)),tf.math.abs(df)), tf.math.reduce_max(tf.math.abs(df)))
        return tf.math.multiply(y_true, upstream), tf.math.multiply(dldy,upstream)
        #return None, tf.math.multiply(dldy,upstream)

    mse = tf.math.divide(tf.math.reduce_sum(tf.math.abs(df)), tf.cast(tf.size(df), tf.float32))
    #mse = tf.math.divide(tf.math.reduce_sum(tf.math.square(df)), tf.cast(tf.size(df), tf.float32))
    psnr = tf.math.multiply(10., tf.math.divide(tf.math.log(tf.math.divide(tf.math.reduce_max(tf.math.square(y_true)), mse)), tf.math.log(10.)))
    #loss = tf.math.subtract(100., psnr)
    loss = tf.math.subtract(100., psnr)

    return loss, grad
'''

@tf.function
def MyLoss(y_true, y_pred):


    #return tf.keras.losses.MSLE(y_true, y_pred)
    #return tf.sqrt(tf.keras.losses.MSE(y_true, y_pred))
    #return tf.keras.losses.MAE(y_true, y_pred)
    #return tf.keras.losses.MSE(y_true, y_pred)
    #return 10./tf.image.psnr(y_true, y_pred, max_val=1.0)
    return (100 - tf.image.psnr(y_true, y_pred, max_val=1.0)) * (1. - (tf.image.ssim(y_true, y_pred, max_val=1.0) + 1.)/2.)
    #return ((100 - tf.image.psnr(y_true, y_pred, max_val=1.0)) + (100. - (100*(tf.image.ssim(y_true, y_pred, max_val=1.0) + 1.)/2.)))/2.
    #return 100. - (100*(tf.image.ssim(y_true, y_pred, max_val=1.0) + 1.)/2.)
    #return (1. - (tf.image.ssim(y_true, y_pred, max_val=1.0) + 1.)/2.)
  


class CustomModel(Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        if tf.equal(l,tf.constant(1, dtype=tf.dtypes.int16)):
            self.optimizer.apply_gradients(zip(gradients / 2., trainable_vars))
        else:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

'''


@tf.function
def augment(image, mask):
    image, mask = tf.cond(tf.equal(tf.random.uniform(shape=[], maxval=2, dtype=tf.int32), tf.constant(1)),
                          lambda: [tf.image.rot90(image), tf.image.rot90(mask)],
                          lambda: [image, mask])
    image, mask = tf.cond(tf.greater(tf.random.uniform(shape=[], dtype=tf.float32), tf.constant(0.5)),
                          lambda: [tf.image.flip_left_right(image), tf.image.flip_left_right(mask)],
                          lambda: [image, mask])
    return image, mask


def build_unet(input_shape=(256, 256, 3), num_classes=3):
    conv2dparam = dict(padding='same', kernel_initializer=keras.initializers.initializers_v1.HeNormal(), kernel_regularizer=l2(1e-04),
                       bias_regularizer=l2(1e-04),
                       activity_regularizer=l2(1e-04))
    conv2dtparam = dict(kernel_size=(2, 2), strides=(2, 2), kernel_initializer=keras.initializers.initializers_v1.HeNormal(),
                        kernel_regularizer=l2(1e-04),
                        bias_regularizer=l2(1e-04),
                        activity_regularizer=l2(1e-04))

    inputs = Input(shape=input_shape)
    # 48
    '''
    down0 = Conv2D(32, (3, 3), **conv2dparam)(inputs)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), **conv2dparam)(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    '''

    down1 = Conv2D(64, (3, 3), **conv2dparam)(inputs)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), **conv2dparam)(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 24

    down2 = Conv2D(128, (3, 3), **conv2dparam)(down1_pool)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), **conv2dparam)(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 12

    down3 = Conv2D(256, (3, 3), **conv2dparam)(down2_pool)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), **conv2dparam)(down3)
    down3 = Activation('relu')(down3)
    down3 = Dropout(0.5)(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 6

    center = Conv2D(512, (3, 3), **conv2dparam)(down3_pool)
    center = Activation('relu')(center)
    center = Conv2D(512, (3, 3), **conv2dparam)(center)
    center = Activation('relu')(center)
    center = Dropout(0.5)(center)
    # center

    # BiasLearnRateFactor = 2 not set
    up3 = Conv2DTranspose(256, **conv2dtparam)(center)
    up3 = Activation('relu')(up3)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), **conv2dparam)(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), **conv2dparam)(up3)
    up3 = Activation('relu')(up3)
    # 12

    up2 = Conv2DTranspose(128, **conv2dtparam)(up3)
    up2 = Activation('relu')(up2)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), **conv2dparam)(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), **conv2dparam)(up2)
    up2 = Activation('relu')(up2)
    # 24

    up1 = Conv2DTranspose(64, **conv2dtparam)(up2)
    up1 = Activation('relu')(up1)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), **conv2dparam)(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), **conv2dparam)(up1)
    up1 = Activation('relu')(up1)

    '''
    up0 = Conv2DTranspose(32, **conv2dtparam)(up1)
    up0 = Activation('relu')(up0)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), **conv2dparam)(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), **conv2dparam)(up0)
    up0 = Activation('relu')(up0)
    '''
    
    #classify = Conv2D(num_classes, (1, 1), **conv2dparam)(up1)

    classify = Conv2D(num_classes, (1, 1), padding='valid', kernel_initializer=keras.initializers.initializers_v1.HeNormal(), kernel_regularizer=l2(1e-04),
                       bias_regularizer=l2(1e-04),
                       activity_regularizer=l2(1e-04))(up1)

    # 48

    model = Model(inputs=inputs, outputs=classify, name='UNET_HOPE')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-04, epsilon=1e-08), metrics=['accuracy', 'MeanSquaredError', tf.keras.metrics.RootMeanSquaredError(name='Rmse')], loss=MyLoss) #loss=tf.keras.losses.MSE)

    model.save('/home/ascalella/dataset/Network/unet.h5')

    return model


def get_unet():
    try:
        model = load_model('/home/ascalella/dataset/Network/unet.h5', custom_objects={'MyLoss': MyLoss})
    except OSError:
        model = build_unet()
    return model


def train_unet(n_epoch, n_img=0, val_split=0):
    image_list = []
    blurred_list = []
    val_image_list = []
    val_blurred_list = []

    dir_img = '/home/ascalella/dataset/Train/Tiles_original'
    dir_blurred = '/home/ascalella/dataset/Train/Tiles_decoded_NEWNEW24'
    sub_dir = listdir(dir_blurred)

    for sub in sub_dir:
        files_img = listdir(dir_blurred + '/{0}'.format(sub))
        if n_img == 0:
            n_img_cla = len(files_img)
        else:
            n_img_cla = n_img/4
        
        for filepath, k in zip(files_img, range(n_img_cla)):
            if val_split != 0 and k >= n_img_cla*(1. - val_split):
                val_image_list.append(imread(dir_img + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
                val_blurred_list.append(imread(dir_blurred + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
            else:
                image_list.append(imread(dir_img + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
                blurred_list.append(imread(dir_blurred + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
        
    image = np.asarray(image_list, dtype='float32')
    blurred = np.asarray(blurred_list, dtype='float32')

    #image = tf.image.convert_image_dtype(image_list[0:750], dtype=tf.float32, saturate=False)
    #blurred = tf.image.convert_image_dtype(blurred_list[0:750], dtype=tf.float32, saturate=False)

    ksizes = [1, 48, 48, 1]
    strides = [1, 48, 46, 1]
    rates = [1, 1, 1, 1]

    pchs_img = tf.image.extract_patches(image/65535., ksizes, strides, rates, 'SAME')
    pchs_blurred = tf.image.extract_patches(blurred/65535. , ksizes, strides, rates, 'SAME')
    pchs_img = tf.reshape(pchs_img, [-1, 48, 48, 3])
    pchs_blurred = tf.reshape(pchs_blurred, [-1, 48, 48, 3])

    image_datagen = tf.data.Dataset.from_tensor_slices(pchs_img)
    blurred_datagen = tf.data.Dataset.from_tensor_slices(pchs_blurred)

    train_generator = tf.data.Dataset.zip((blurred_datagen, image_datagen))
    train_generator = train_generator.map(augment).batch(132)

    val_generator = None
    if val_split!=0:
        val_image = np.asarray(val_image_list, dtype='float32')
        val_blurred = np.asarray(val_blurred_list, dtype='float32')

        val_pchs_img = tf.image.extract_patches(val_image/65535., ksizes, strides, rates, 'SAME')
        val_pchs_blurred = tf.image.extract_patches(val_blurred/65535. , ksizes, strides, rates, 'SAME')
        val_pchs_img = tf.reshape(val_pchs_img, [-1, 48, 48, 3])
        val_pchs_blurred = tf.reshape(val_pchs_blurred, [-1, 48, 48, 3])

        val_image_datagen = tf.data.Dataset.from_tensor_slices(val_pchs_img)
        val_blurred_datagen = tf.data.Dataset.from_tensor_slices(val_pchs_blurred)

        val_generator = tf.data.Dataset.zip((val_blurred_datagen, val_image_datagen))
        val_generator = val_generator.map(augment).batch(132)

    model = get_unet()

    checkpoint = ModelCheckpoint('/home/ascalella/dataset/Network/Checkpoint/unet_{epoch:04d}.h5', monitor='loss', save_best_only=False, mode='auto', period=10)
    hs = model.fit(train_generator, epochs=n_epoch, callbacks=[checkpoint],  validation_data=val_generator) #steps_per_epoch=132

    #model.save('/home/ascalella/dataset/Network/unet.h5')

    pkl.dump(hs.history, open('/home/ascalella/dataset/Network/Checkpoint/UNETtrain_history.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)

    return model, hs, train_generator


def train_unet2(n_epoch, n_img=0, val_split=0):
    image_list = []
    blurred_list = []
    val_image_list = []
    val_blurred_list = []

    dir_img = '/home/ascalella/dataset/Train/Tiles_original'
    dir_blurred = '/home/ascalella/dataset/Train/Tiles_decoded_Filter24'
    sub_dir = listdir(dir_blurred)

    for sub in sub_dir:
        files_img = listdir(dir_blurred + '/{0}'.format(sub))
        if n_img == 0:
            n_img_cla = len(files_img)
        else:
            n_img_cla = n_img/4
        
        for filepath, k in zip(files_img, range(n_img_cla)):
            if val_split != 0 and k >= n_img_cla*(1. - val_split):
                val_image_list.append(imread(dir_img + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
                val_blurred_list.append(imread(dir_blurred + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
            else:
                image_list.append(imread(dir_img + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
                blurred_list.append(imread(dir_blurred + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
        
    image = np.asarray(image_list, dtype='float32')
    blurred = np.asarray(blurred_list, dtype='float32')

    #image = tf.image.convert_image_dtype(image_list[0:750], dtype=tf.float32, saturate=False)
    #blurred = tf.image.convert_image_dtype(blurred_list[0:750], dtype=tf.float32, saturate=False)

    ksizes = [1, 256, 256, 1]
    strides = [1, 256, 256, 1]
    rates = [1, 1, 1, 1]

    pchs_img = tf.image.extract_patches(image/65535., ksizes, strides, rates, 'SAME')
    pchs_blurred = tf.image.extract_patches(blurred/65535. , ksizes, strides, rates, 'SAME')
    pchs_img = tf.reshape(pchs_img, [-1, 256, 256, 3])
    pchs_blurred = tf.reshape(pchs_blurred, [-1, 256, 256, 3])

    image_datagen = tf.data.Dataset.from_tensor_slices(pchs_img)
    blurred_datagen = tf.data.Dataset.from_tensor_slices(pchs_blurred)

    train_generator = tf.data.Dataset.zip((blurred_datagen, image_datagen))
    train_generator = train_generator.map(augment).batch(4)

    val_generator = None
    if val_split!=0:
        val_image = np.asarray(val_image_list, dtype='float32')
        val_blurred = np.asarray(val_blurred_list, dtype='float32')

        val_pchs_img = tf.image.extract_patches(val_image/65535., ksizes, strides, rates, 'SAME')
        val_pchs_blurred = tf.image.extract_patches(val_blurred/65535. , ksizes, strides, rates, 'SAME')
        val_pchs_img = tf.reshape(val_pchs_img, [-1, 48, 48, 3])
        val_pchs_blurred = tf.reshape(val_pchs_blurred, [-1, 48, 48, 3])

        val_image_datagen = tf.data.Dataset.from_tensor_slices(val_pchs_img)
        val_blurred_datagen = tf.data.Dataset.from_tensor_slices(val_pchs_blurred)

        val_generator = tf.data.Dataset.zip((val_blurred_datagen, val_image_datagen))
        val_generator = val_generator.map(augment).batch(64)

    model = get_unet()

    checkpoint = ModelCheckpoint('/home/ascalella/dataset/Network/Checkpoint_LAST/unet_{epoch:04d}.h5', monitor='loss', save_best_only=False, mode='auto', period=10)
    hs = model.fit(train_generator, epochs=n_epoch, initial_epoch=30, callbacks=[checkpoint],  validation_data=val_generator) #steps_per_epoch=132

    #model.save('/home/ascalella/dataset/Network/unet.h5')

    pkl.dump(hs.history, open('/home/ascalella/dataset/Network/Checkpoint_LAST/UNETtrain_history.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)

    return model, hs, train_generator