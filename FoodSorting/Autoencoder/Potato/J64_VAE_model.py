import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization, ReLU, LeakyReLU
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from keras.activations import relu
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K                         #contains calls for tensor manipulations

#################################
#       Hyperparameters         #
#################################

latent_dim = 64

# the num_channels parameter can be configured to equal the number of image channels (grayscale data so == 1 (RGB==3))
img_height, img_width, num_channels = 128, 128, 1
input_shape = (img_height, img_width, num_channels)

# verbosity mode is set to True (by means of 1), which means that all the output is shown on screen.
verbosity = 1 

alpha = 0.1 #0.2                    # LeakyReLU (alpha=0.01 normally)

kernel_size1 = 3                #(5,5)
kernel_size2 = 3                #(3,3)
strides1 = 1                     #(1,1)
strides2 = 2                     #(2,2)

filter1 = 32
filter2 = 32 #filter1*2 #32
filter3 = 32 #filter1*4 #32
filter4 = 32 #filter1*8 #32
filter5 = 32 #filter1*16 #32

intermediate_dim = 200 # 100, 200



#################################
#           Encoder             #
#################################

"""
Use the sampling with a Lambda to ensure that correct gradients are computed during 
the backwards pass based on our values for mu and log sigma
"""
# Define sampling with reparameterization trick
def sample_z(args):
    """
    instead of sampling from q(z|x), use reparameterization trick and sample from
    epsilon = N(0,1), then z = z_mean + sqrt(var)*eps

    z_mean = latent_mean
    sigma = latent_sigma
    """
    z_mean, z_log_var = args
    batch     = K.shape(z_mean)[0]
    dim       = K.int_shape(z_mean)[1]
    eps       = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(z_log_var / 2) * eps

# ================= #
#     Encoder       #
# ================= #

# Image inputs:
i       = Input(shape=input_shape, name='encoder_input')

# First 2D convolution layer with batch normalization, max pooling and dropout:
cx      = Conv2D(filters=filter1, kernel_size=kernel_size1, strides=strides2, padding='same')(i)
cx      = LeakyReLU(alpha=alpha)(cx)

# Second 2D convolution:
cx      = Conv2D(filters=filter2, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx      = LeakyReLU(alpha=alpha)(cx)

# Third 2D convolution:
cx      = Conv2D(filters=filter3, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx      = LeakyReLU(alpha=alpha)(cx)

# Fourth 2D convolution:
cx      = Conv2D(filters=filter4, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx      = LeakyReLU(alpha=alpha)(cx)

# Fifth 2D convolution:
cx      = Conv2D(filters=filter5, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx      = LeakyReLU(alpha=alpha)(cx)

# Get Conv2D shape for Conv2DTranspose operation in decoder (retrieve the shape of the final Conv2D output)
conv_shape = K.int_shape(cx)

# Prepare a vector for the fully connected layers (reshapes data to be suitable for dense layer):
x       = Flatten()(cx)

# Add one fully-connected layer:
x       = Dense(units=intermediate_dim, name="latent")(x)
x       = LeakyReLU(alpha=alpha)(x)

# Get mean and std of the above latent space (bottleneck layer);
z_mean      = Dense(latent_dim, name='z_mean')(x)
z_log_var   = Dense(latent_dim, name='z_log_mean')(x)

# Use reparameterization trick to ensure correct gradient (wrap z as a layer)
z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mean, z_log_var])

# Instantiate encoder
encoder = Model(i, [z_mean, z_log_var, z], name='encoder')




#################################
#           Decoder             #
#################################

# ================= #
#     Decoder       #
# ================= #

# Latent inputs:
d_i   = Input(shape=(latent_dim, ), name='decoder_input')

# upsamples data for deconvolutional layers:
x     = Dense(units = conv_shape[1] * conv_shape[2] * conv_shape[3], activation='tanh')(d_i)

# reshapes data to be suitable for deconvolutional layers:
x     = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)

# first transposed 2D convolutional (deconvolution) layer:
cx    = Conv2DTranspose(filters=filter5, kernel_size=kernel_size1, strides=strides2, padding='same')(x)
cx    = ReLU()(cx)

# second transposed 2D convolutional (deconvolution) layer:
cx    = Conv2DTranspose(filters=filter5, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx    = ReLU()(cx)

# third transposed 2D convolutional (deconvolution) layer:
cx    = Conv2DTranspose(filters=filter5, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx    = ReLU()(cx)

# fourth transposed 2D convolutional (deconvolution) layer:
cx    = Conv2DTranspose(filters=filter5, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx    = ReLU()(cx)

# fifth transposed 2D convolutional (deconvolution) layer: 
cx    = Conv2DTranspose(filters=filter5, kernel_size=kernel_size1, strides=strides2, padding='same')(cx)
cx    = ReLU()(cx)

# use sigmoid activation function for output image:
o     = Conv2DTranspose(filters=num_channels, kernel_size=kernel_size1, activation='sigmoid', padding='same', 
                        name='decoder_output')(cx)     ### softplus is a strong candidate against sigmoid...

# Instantiate decoder
decoder = Model(d_i, o, name='decoder')




#################################
#            VAE                #
#################################

# ================= #
#  VAE as a whole   #
# ================= #

# Instantiate VAE
vae_outputs = decoder(encoder(i)[2])
vae         = Model(i, vae_outputs, name='vae')


#################################
#     Loss & Cost Functions     #
#################################
# Define loss (customized loss layer)
def kl_reconstruction_loss(true, pred):
    """
    Loss for each mini-batch.
    
    true = before (orignial image)
    pred = after  (reconstructed image)
    """
    # Reconstruction loss given a Bernoulli likelihood (used for binary inputs)
    #reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height * num_channels
    
    # Reconstruction loss given a Gaussian likelihood
    reconstruction_loss = mse(K.flatten(true), K.flatten(pred)) 
    reconstruction_loss *= img_width * img_height * num_channels  
    #-----> reconstruction_loss = np.mean(np.square(true - pred, axis=1)) * img_width * img_height * num_channels 
    
    # KL divergence loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # Total loss = 50% reconstruction loss + 50% KL divergence loss
    return reconstruction_loss + kl_loss

# Cost Function: average over minibatches:
def vae_cost_function(true, pred):
    loss_function = kl_reconstruction_loss(true, pred)
    return tf.reduce_mean(loss_function)      # K.mean(loss_function)



#################################
#     Initialized model         #
#################################
### Create initialized model
#Roundabount for training on GPU: `add_loss method`.

def create_model_lr():
    """
    model instance used for the learning rate finder
    """
    model = vae
    model.add_loss(vae_cost_function(i, vae_outputs))   # add loss to model
    
    # ADAM
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)  # ADAM lr parameters in adam() optimizer

    # Compile VAE
    model.compile(optimizer=optimizer, loss=None)     # No need to pass any loss function to compile method
    return model


def create_model():
    """
    VAE model instanced with optimal choosen learning rate alpha
    """
    model = vae
    model.add_loss(vae_cost_function(i, vae_outputs))   # add loss to model
    
    # ADAM
    alpha = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999)

    # Compile VAE
    model.compile(optimizer=optimizer, loss=None)     # No need to pass any loss function to compile method
    return model