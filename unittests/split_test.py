import unittest
from datetime import timedelta
import pandas as pd
from config import ROOT_DIR

data_path = "data"


class TrainTestSplitTestCase(unittest.TestCase):
    def test_train_test_split(self):
        from dataprocessing.process import TrainTestSplit
        train_test_split = TrainTestSplit()

        train_features = pd.read_pickle(ROOT_DIR / data_path / "train_features.pkl")
        train_outcomes = pd.read_pickle(ROOT_DIR / data_path / "train_outcomes.pkl")

        # X_train, X_val, y_train, y_val = train_test_split(train_features, train_outcomes)
        train_test_split(train_features, train_outcomes)
        ratio = train_test_split.ratio

        print(f"train / test ratio: {ratio[0]} / {ratio[1]}")
        self.assertGreaterEqual(train_test_split.val_start_date - train_test_split.train_end_date, timedelta(days=365))


if __name__ == '__main__':
    unittest.main()
