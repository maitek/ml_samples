import numpy as np

class CsvDataset():
    """ CSV data set """

    def __init__(self, csv_file, delimiter=';', target_identifier = 't', train=True):
        """
        Args:
            csv_file (string): Path to the csv file
            train (bool): True if training set, else false
        """

        with open(csv_file) as f:
            header = f.readline()

        col_names = np.array(header.strip().split(delimiter))

        self.data = np.genfromtxt(csv_file, delimiter=delimiter, skip_header=1, dtype=np.float32)

        # use 90% for train and 10% for test
        train_data, test_data = np.split(self.data, [int(self.data.shape[0]*0.9)])
        self.data = train_data if train else test_data

        self.data, self.targets = self.data[:,col_names != target_identifier], self.data[:,col_names == target_identifier]

        # normalize data
        self.mu = np.mean(self.data)
        self.sigma = np.std(self.data)
        self.data = (self.data - self.mu) / self.sigma

        self.num_targets = self.targets.shape[1]
        self.num_features = self.data.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X, y = self.data[idx,:], self.targets[idx,:]
        return X
