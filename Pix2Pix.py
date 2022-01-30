import tensorflow as tf
import numpy as np
import datetime
from os import listdir
from skimage.io import imread
from tensorflow.keras import backend




OUTPUT_CHANNELS = 3
LAMBDA = 100

#@tf.custom_gradient
@tf.function
def MyLoss(y_true, y_pred):


    #return tf.keras.losses.MSLE(y_true, y_pred)
    #return tf.sqrt(tf.keras.losses.MSE(y_true, y_pred))
    #return tf.keras.losses.MAE(y_true, y_pred)
    #return tf.keras.losses.MSE(y_true, y_pred)
    #return 10./tf.image.psnr(y_true, y_pred, max_val=2.0)
    #return 100/tf.image.psnr(y_true, y_pred, max_val=2.0) * (1. - (tf.image.ssim(y_true, y_pred, max_val=2.0) + 1.)/2.)
    #return (1. - (tf.image.ssim(y_true, y_pred, max_val=2.0) + 1.)/2.)
    return (100 - tf.image.psnr(y_true, y_pred, max_val=2.0)) * (1. - (tf.image.ssim(y_true, y_pred, max_val=2.0) + 1.)/2.)

    '''
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

    mse = tf.math.divide(tf.math.reduce_sum(tf.math.square(df)), tf.cast(tf.size(df), tf.float32))
    psnr = tf.math.multiply(10., tf.math.divide(tf.math.log(tf.math.divide(tf.math.reduce_max(tf.math.square(y_true)), mse)), tf.math.log(10.)))
    loss = tf.math.subtract(100., psnr)

    return loss, grad
    '''

def get_pix():
    try:
        model = tf.keras.models.load_model('/home/ascalella/dataset/Network/generator.h5', custom_objects={'MyLoss': MyLoss})
    except OSError:
        model = Generator()
    return model

def get_disc():
    try:
        model = tf.keras.models.load_model('/home/ascalella/dataset/Network/discriminator.h5')
    except OSError:
        model = Discriminator()
    return model


@tf.function
def augment(image, mask):
    image, mask = tf.cond(tf.equal(tf.random.uniform(shape=[], maxval=2, dtype=tf.int32), tf.constant(1)),
                          lambda: [tf.image.rot90(image), tf.image.rot90(mask)],
                          lambda: [image, mask])
    image, mask = tf.cond(tf.greater(tf.random.uniform(shape=[], dtype=tf.float32), tf.constant(0.5)),
                          lambda: [tf.image.flip_left_right(image), tf.image.flip_left_right(mask)],
                          lambda: [image, mask])
    return image, mask


def generator_loss(disc_generated_output, gen_output, target, loss_object):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = MyLoss(target, gen_output) #tf.reduce_mean(tf.abs(target - gen_output)) 

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(loss=MyLoss)

    return model;


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    model = tf.keras.Model(inputs=[inp, tar], outputs=last)

    model.compile()

    return model


def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


@tf.function
def train_step(input_image, target, epoch, generator, discriminator, loss_object, summary_writer, generator_optimizer,discriminator_optimizer):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target,
                                                                   loss_object)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    #with summary_writer.as_default():
        #tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        #tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        #tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        #tf.summary.scalar('disc_loss', disc_loss, step=epoch)


    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def fit(train_ds, epochs):
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    generator = get_pix()
    discriminator = get_disc()
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    log_dir = '/home/ascalella/dataset/Train_Result/Log/'

    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    a = 0
    b = 0
    c = 0
    d = 0

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        # Train
        for n, (input_image, target) in train_ds.enumerate():
            tf.print('.', end='')
            if n % 50 == 0:
                tf.print(n)
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, epoch, generator, discriminator, loss_object, summary_writer, generator_optimizer, discriminator_optimizer)
            
            a += gen_total_loss
            b += gen_gan_loss
            c += gen_l1_loss
            d += disc_loss
        tf.print('gen_total_loss', a/float(n))
        tf.print('gen_gan_loss', b/float(n))
        tf.print('gen_l1_loss', c/float(n))
        tf.print('disc_loss', d/float(n))

        a = 0
        b = 0
        c = 0
        d = 0

        if np.remainder(epoch + 1,10) == 0:
            generator.save('/home/ascalella/dataset/Network/Checkpoint_MYPIX24/generator' + str(epoch+1) + '.h5')
            discriminator.save('/home/ascalella/dataset/Network/Checkpoint_MYPIX24/discriminator' + str(epoch+1) + '.h5')
        


def train_pix(n_epoch, n_img=0):
    image_list = []
    blurred_list = []

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
            image_list.append(imread(dir_img + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))
            blurred_list.append(imread(dir_blurred + '/{0}'.format(sub) + '/{0}'.format(filepath), plugin='tifffile'))

    image = np.asarray(image_list, dtype='float32')
    blurred = np.asarray(blurred_list, dtype='float32')

    ksizes = [1, 256, 256, 1]
    strides = [1, 256, 256, 1]
    rates = [1, 1, 1, 1]
    pchs_img = tf.image.extract_patches((image / 32767.5) - 1., ksizes, strides, rates, 'SAME')
    pchs_blurred = tf.image.extract_patches((blurred / 32767.5) - 1., ksizes, strides, rates, 'SAME')

    pchs_img = tf.reshape(pchs_img, [-1, 256, 256, 3])
    pchs_blurred = tf.reshape(pchs_blurred, [-1, 256, 256, 3])

    image_datagen = tf.data.Dataset.from_tensor_slices(pchs_img)
    blurred_datagen = tf.data.Dataset.from_tensor_slices(pchs_blurred)

    train_generator = tf.data.Dataset.zip((blurred_datagen, image_datagen))

    train_generator = train_generator.map(augment).batch(4)

    fit(train_generator, n_epoch)
