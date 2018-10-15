from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import data_prep
from nn import *
from keras.layers import Conv1D, Conv2D, Dense, MaxPooling1D, Flatten, Embedding, Reshape, MaxPool2D, LSTM
# from keras import backend as K
# from rnn import w_categorical_crossentropy, loss_class_accuracy
# from functools import partial, update_wrapper

class CNNLSTM:
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim

        self.model = self.__prepare_model()

    def __prepare_model(self):

        print('Build model...')
        # w_array = np.ones((3, 3))
        # penalty = 1.2
        # w_array[2, 1] = penalty
        # w_array[1, 2] = penalty
        # custom_loss = partial(w_categorical_crossentropy, weights=w_array)
        # custom_loss.__name__ = 'w_categorical_crossentropy'

        # convolutional layer
        self.nb_filter = 50 # 50 output channels
        self.kernel_size_x = self.input_shape[0]//10 # 50 -> 5
        self.kernel_size_y = self.input_shape[1] # 48

        model = Sequential()
        model.add(Conv2D(self.nb_filter, (self.kernel_size_x, self.kernel_size_y), input_shape=self.input_shape, activation='relu'))
        # model.add(MaxPool2D(pool_size=(2, 2))) by imposing kernel_size_y to #features
        model.add(Dropout(0.2))
        model.add(Reshape(target_shape=(50-self.kernel_size_x+1, self.nb_filter)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        print('Compile model...')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def get_model(self):
        return self.model


def main():
    timestep = 50
    n_cross_validation = 3

    data = data_prep.get_test_data(timestep, predict_step=5, filename="upbit_l2_orderbook_ADA")
    # for convolutional layer consider sample as a 2d gray picture
    # x.shape[1] time step(input_length)
    # x.shape[2] features(input_dim)

    # data.x.shape <- (n_training_samples, timestep, 1, n_features)
    data.x = data.x.reshape(data.x.shape[0], data.x.shape[1], data.x.shape[2], 1).astype('float32')
    nn = NeuralNetwork(CNNLSTM(input_shape = data.x.shape[1:], output_dim= data.y.shape[1]), class_weight= {0:1., 1:1., 2:1.}) # num of output dim

    print("TRAIN")
    nn.train(data)

    print("TEST")
    nn.test(data)

    print("TRAIN WITH CROSS-VALIDATION")
    # nn.run_with_cross_validation(data, n_cross_validation)

if __name__ == '__main__':
    main()
