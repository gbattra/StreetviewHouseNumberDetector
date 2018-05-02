import numpy as np
from services.data_preprocessor_service import DataPreprocessorService as dps


class PredictionHelper:

    @staticmethod
    def predict(Z, num_classes):
        return dps.one_hot_encode(Z.argmax(axis=1), num_classes)
