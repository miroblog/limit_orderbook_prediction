import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

class NeuralNetwork:
    def __init__(self, nn, class_weight, validation_split=0.25, batch_size=256, nb_epoch=200, show_accuracy=True):
        self.show_accuracy = show_accuracy
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.nn = nn
        self.model = nn.get_model()
        # unbalanced weight
        # past 1 : 10 : 1, inverse of ration is preferred
        self.class_weight = class_weight

    def train(self, data):
        print("Train...")
        filepath = "./model/improvement-{epoch:02d}-{val_loss_class_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss_class_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        print("before fit")
        self.model.fit(data.x,
                       data.y,
                       epochs=self.nb_epoch,
                       batch_size=self.batch_size,
                       validation_split=self.validation_split,
                       verbose=1,
                       callbacks=callbacks_list,
                       class_weight=self.class_weight)
        return self.test(data)

    def test(self, data):
        y_predict_proba = self.model.predict(data.x)
        y_predicted = y_predict_proba.argmax(axis=-1)
        y_actual_proba = data.y
        y_actual = y_actual_proba.argmax(axis=-1)
        error = (np.ravel(y_predicted) != np.ravel(y_actual)).sum().astype(float)/y_actual.shape[0]
        cmat = confusion_matrix(y_actual, y_predicted)
        class_num = cmat.diagonal()
        class_accuracy = class_num / cmat.sum(axis=1)
        print("EACH : class 0: {0}, class 1: {1}, class 2: {2}".format(
            class_accuracy[0],
            class_accuracy[1],
            class_accuracy[2]
        ))
        print("PREDICTED: class 0: {0}, class 1: {1}, class 2: {2}".format(
            class_num[0],
            class_num[1],
            class_num[2]))
        print("ACTUAL: class 0: {0}, class 1: {1}, class 2: {2}".format(
              np.sum(np.ravel(y_actual) == 0),
              np.sum(np.ravel(y_actual) == 1),
              np.sum(np.ravel(y_actual) == 2)))
        print("ERROR RATE: ", error)

        return error

    # def run_with_cross_validation(self, data, cross_num):
    #     return self.__run_with_cross_validation(data.x, data.y, cross_num)
    #
    # def __run_with_cross_validation(self, x, y, cross_num):
    #     # N - number of observations
    #     N = len(x)
    #     train_errors = np.zeros(cross_num)
    #     test_errors = np.zeros(cross_num)
    #     cv = cross_validation.KFold(N, cross_num, shuffle=True)
    #
    #     i = 0
    #     for train_index, test_index in cv:
    #         x_train = x[train_index, :]
    #         y_train = y[train_index, :]
    #         x_test = x[test_index, :]
    #         y_test = y[test_index, :]
    #
    #         train_data = Data(x_train, y_train)
    #         test_data = Data(x_test, y_test)
    #
    #         train_errors[i] = self.train(train_data)
    #         test_errors[i] = self.test(test_data)
    #         i += 1
    #
    #     return train_errors, test_errors

    def __change_input_dim(self, input_dim):
        self.nn.change_input_dim(input_dim)
        self.model = self.nn.get_model()

    def feature_selection(self, data, cross_val_passes=3):
        # N - number of observations, T - number of time points, M - number of features
        N, T, M = data.x.shape
        print("M = {0}".format(M))
        best_err_rate = 1
        best_feature = 0
        available_features = range(M)
        selected_features = []
        temp_selected_features = []

        results = []

        for i in range(M):
            progress = False
            self.__change_input_dim(i+1)
            for feature in available_features:
                current_feature_set = list(selected_features)
                current_feature_set.append(feature)
                print("current feature set = {0}".format(current_feature_set))

                train_errors, test_errors = self.__run_with_cross_validation(
                        data.x[:, :, current_feature_set],
                        data.y,
                        cross_val_passes)

                error = np.average(test_errors)
                results.append((current_feature_set, error))

                if error < best_err_rate:
                    best_err_rate = error
                    best_feature = feature
                    temp_selected_features = list(current_feature_set)
                    progress = True
            if not progress:
                break
            available_features.remove(best_feature)
            selected_features = temp_selected_features
            print("selected features = {0}, err reate = {1}".format(selected_features, best_err_rate))

        return selected_features, results
