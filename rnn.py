from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from nn import *
from keras.optimizers import SGD
import data_prep
from keras.models import model_from_json
from itertools import product
from functools import partial, update_wrapper
import time

from keras import backend as K


# INTERESTING_CLASS_ID = 2  # Choose the class of interest
#
# def w_categorical_crossentropy(y_true, y_pred, weights):
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
#     y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#         final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
#     cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
#     return cross_ent * final_mask
#
# def loss_class_accuracy(y_true, y_pred):
#     class_id_true = K.argmax(y_true, axis=-1)
#     class_id_preds = K.argmax(y_pred, axis=-1)
#     # Replace class_id_preds with class_id_true for recall here
#     accuracy_mask = K.cast(K.equal(class_id_preds, INTERESTING_CLASS_ID), 'int32')
#     class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
#     class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
#     return class_acc

class RNN:
    def __init__(self, input_shape, output_dim):
        self.input_length, self.input_dim = input_shape[0], input_shape[1]
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

        model = Sequential()
        model.add(LSTM(64, return_sequences=True,
                       input_shape=(self.input_length, self.input_dim)))
        model.add(LSTM(64, return_sequences=False, input_shape=(self.input_length, self.input_dim)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(self.output_dim, activation='softmax'))

        print('Compile model.   ..')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def get_model(self):
        return self.model


def main():

    timestep = 50
    n_cross_validation = 3
    # for order book info only
    data = data_prep.get_test_data(timestep, predict_step=5, filename="upbit_l2_orderbook_ADA")

    # input_shape <- (timestep, n_features)
    # output_shape <- n_classes
    nn = NeuralNetwork(RNN(input_shape=data.x.shape[1:], output_dim=data.y.shape[1]), class_weight= {0:1., 1:1., 2:1.})

    print("TRAIN")
    nn.train(data)

    print("TEST")
    nn.test(data)

    # print("TRAIN WITH CROSS-VALIDATION")
    # nn.run_with_cross_validation(data, n_cross_validation)

if __name__ == '__main__':
    main()
