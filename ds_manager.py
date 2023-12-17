import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DSManager:
    def __init__(self,reduced_features=True, reduced_rows=True):
        np.random.seed(0)
        dataset = "data/dataset_66_min.csv"
        self.reduced_features = reduced_features
        self.reduced_rows = reduced_rows
        if reduced_features:
            if reduced_rows:
                dataset = "data/dataset_66_min.csv"
            else:
                dataset = "data/dataset_66.csv"
        else:
            if reduced_rows:
                dataset = "data/dataset_min.csv"
            else:
                dataset = "data/dataset.csv"
        df = pd.read_csv(dataset)
        self.X_columns = DSManager.get_wavelengths(reduced_features)
        self.y_column = "oc"
        df = df[self.X_columns+[self.y_column]]
        df = df.sample(frac=1)
        self.full_data = df.to_numpy()
        self.full_data = DSManager._normalize(self.full_data)

    def __repr__(self):
        return self.get_name()

    def get_name(self):
        if self.reduced_features:
            if self.reduced_rows:
                return "Reduced Rows & Features"
            else:
                return "All Rows, Reduced Features"
        else:
            if self.reduced_rows:
                return "Reduced Rows, All Features"
            else:
                return "All Rows & Features"

    def count_rows(self):
        return self.full_data.shape[0]

    def count_features(self):
        return len(self.X_columns)

    @staticmethod
    def wavelengths_itr():
        wvs = []
        spec = 400
        while spec <= 2499.5:
            n_spec = spec
            if int(n_spec) == spec:
                n_spec = int(n_spec)
            wavelength = str(n_spec)
            yield wavelength
            spec = spec + 0.5
        return wvs

    @staticmethod
    def get_wavelengths(reduced_features=True):
        if reduced_features:
            return [str(i) for i in range(66)]
        return list(DSManager.wavelengths_itr())

    @staticmethod
    def _normalize(data):
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        return data

    def get_datasets(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        validation_x = validation_data[:, :-1]
        validation_y = validation_data[:, -1]

        return train_x, train_y, test_x, test_y, validation_x, validation_y

    def get_X_y(self):
        return self.get_X_y_from_data(self.full_data)

    @staticmethod
    def get_X_y_from_data(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    def get_train_test_validation(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
        return train_data, test_data, validation_data

    def get_train_test_validation_X_y(self):
        train_data, test_data, validation_data = self.get_train_test_validation()
        return *DSManager.get_X_y_from_data(train_data), \
            *DSManager.get_X_y_from_data(test_data),\
            *DSManager.get_X_y_from_data(validation_data)

    def get_train_test(self):
        train_data, test_data = model_selection.train_test_split(self.full_data, test_size=0.1, random_state=2)
        return train_data, test_data

    def get_train_test_X_y(self):
        train_data, test_data = self.get_train_test()
        return *DSManager.get_X_y_from_data(train_data), \
            *DSManager.get_X_y_from_data(test_data)


if __name__ == "__main__":
    d = DSManager(True, False)
    print(d.full_data.shape)
    # x = d.full_data
    # sampled_rows_indices = np.random.choice(x.shape[0], size=10, replace=False)
    # sampled_rows = x[sampled_rows_indices, :]
    # print(sampled_rows.shape)