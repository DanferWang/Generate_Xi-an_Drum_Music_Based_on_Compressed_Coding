import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Dense, LSTM, BatchNormalization, Flatten, Input, Lambda, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.losses import mean_squared_error
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model

# data
## load data
### n is the total nunmber for the dataset.txt file, totally 1
n = 1
N = n + 1
### set the directory of  the dataset
### pay attention to the name of the dataset
inputDir = '../'
fileName = inputDir + 'zero-oneNorm_dataset.txt'  # datasetName
### output initalize
rePut1 = []
rePut2 = []
rePut3 = []
for i in range(1, N):
    with open(fileName, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            dat1t, dat2t, dat3t = [float(i) for i in lines.split()]
            rePut1.append(dat1t)
            rePut2.append(dat2t)
            rePut3.append(dat3t)
            pass
        pass
### transfer list to array
rePutN1 = np.array(rePut1)  # pitch
rePutN2 = np.array(rePut2)  # duration
rePutN3 = np.array(rePut3)  # velocity
print("number of line: " + str(len(rePutN1)))

## prepare data
### split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

rePutN1 = rePutN1.reshape((len(rePutN1), 1))
rePutN2 = rePutN2.reshape((len(rePutN2), 1))
rePutN3 = rePutN3.reshape((len(rePutN3), 1))
dataset = np.hstack((rePutN1, rePutN2, rePutN3))
pitch_size = len(set(rePut1))
### choose a number of time steps
n_steps = 3
X, y = split_sequences(dataset, n_steps)
### padding matrix
X = pad_sequences(X, maxlen=n_steps, padding='post')
print("data load and prepared successfully!")


# model
## model parameter
embedding_dropout_rate = 0.5
latent_dim = 2
batch_size = 512
epochs_num = 50
validation_split = 0.2
n_features = 3
## encoder
encoder_input = Input(shape=(n_steps,n_features), name="encoder_input")
lstm_1 = LSTM(80, activation="relu", return_sequences=True)(encoder_input)
lstm_1 = BatchNormalization()(lstm_1)
lstm_2 = LSTM(40, activation="relu", return_sequences=True)(lstm_1)
lstm_2 = BatchNormalization()(lstm_2)
fc = Flatten()(lstm_2)
fc = Dense(20, activation="relu")(fc)
fc = BatchNormalization()(fc)
mean = Dense(latent_dim, name="latent_mean")(fc)
stddev = Dense(latent_dim, name="latent_stddev")(fc)
### record shape of lstmed matrix
lstm_shape = K.int_shape(lstm_2)


### get sample z
def sample_z(param):
    mean, stddev = param
    batch = K.shape(mean)[0]
    dim = K.int_shape(mean)[1]
    eps = K.random_normal(shape=(batch, dim))
    z = mean + K.exp(stddev / 2) * eps
    return z


z = Lambda(sample_z, output_shape=(latent_dim,), name="z")([mean, stddev])

### encoder model
encoder = Model(encoder_input, [mean, stddev, z], name="encoder")
encoder.summary()
plot_model(encoder, to_file="VAE_encoder_plot_for_oringinal.png", show_shapes=True, show_layer_names=True)

## decoder
decoder_input = Input(shape=(latent_dim,), name="decoder_input")
x = Dense(lstm_shape[1] * lstm_shape[2], activation="relu")(decoder_input)
x = BatchNormalization()(x)
x = Reshape((lstm_shape[1], lstm_shape[2]))(x)
lstmT_1 = LSTM(80, activation="relu", return_sequences=True)(x)
lstmT_1 = BatchNormalization()(lstmT_1)
lstmT_2 = LSTM(160, activation="relu", return_sequences=True)(lstmT_1)
lstmT_2 = BatchNormalization()(lstmT_2)
decoder_output = Dense(3, activation="relu", name="decoder_output")(lstmT_2)

### decoder model
decoder = Model(decoder_input, decoder_output, name="decoder")
decoder.summary()
plot_model(decoder, to_file="VAE_decoder_plot_for_original.png", show_shapes=True, show_layer_names=True)

## VAE
vae_output = decoder(encoder(encoder_input)[2])
vae = Model(encoder_input, vae_output, name="VAE")
vae.summary()
plot_model(decoder, to_file="VAE_plot_for_original.png", show_shapes=True, show_layer_names=True)


## define loss function with KL divergence
def kl_divergence_loss(true, pred):
    ### 用sparse交叉熵，可以不用事先将类别转换为one hot形式
    u_loss = mean_squared_error(K.flatten(true), K.flatten(pred)) * batch_size * n_steps
    kl_loss = 1 + stddev - K.square(mean) - K.exp(stddev)
    l_loss = K.sum(kl_loss, axis=-1)
    l_loss *= -0.5
    loss = K.mean(u_loss + l_loss)
    return loss


## compile & train the model
vae.compile(optimizer="adam", loss=kl_divergence_loss, metrics=['accuracy'])
his = vae.fit(X, X, epochs=epochs_num, batch_size=batch_size, validation_split=validation_split)
vae_tul = 0

# visualization
plt.plot(his.history["accuracy"])
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()
plt.plot(his.history["loss"])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
