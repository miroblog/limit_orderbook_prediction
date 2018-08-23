from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import data_prep
from nn import *
from keras.layers import Conv1D, Conv2D, Dense, MaxPooling1D, Flatten, Embedding, Reshape, MaxPool2D

class CNNLSTM:
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape[1:]
        self.output_dim = output_dim

        self.model = self.__prepare_model()

    def __prepare_model(self):
        print(self.nb_filter)
        print('Build model...')

        # convolutional layer
        self.nb_filter = 18 #5
        self.kernel_size = 2
        self.filter_length = 4

        model = Sequential()
        model.add(Conv2D(self.nb_filter, (self.kernel_size, self.kernel_size), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        print('Compile model...')
        model.compile(loss='categorical_crossentropy', optimizer='adam')
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
    data.x = data.x.reshape(data.x.shape[0], data.x.shape[1], 1, data.x.shape[2]).astype('float32')
    nn = NeuralNetwork(CNNLSTM(input_shape = data.x.shape[1:], output_dim= data.y.shape[1]), class_weight= {0:1., 1:1., 2:1.}) # num of output dim

    print("TRAIN")
    nn.train(data)

    print("TEST")
    nn.test(data)

    print("TRAIN WITH CROSS-VALIDATION")
    nn.run_with_cross_validation(data, n_cross_validation)

if __name__ == '__main__':
    main()