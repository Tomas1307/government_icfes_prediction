from sklearn.preprocessing import LabelEncoder
import numpy as np

class LabelEncoderWithUnknown(LabelEncoder):
    """
    Versión extendida de LabelEncoder que agrega una clase "__unknown__" para manejar
    valores no vistos durante la predicción.
    """
    def fit(self, y):
        y = y.astype(str)
        classes = np.unique(y)
        if "__unknown__" not in classes:
            classes = np.append(classes, "__unknown__")
        super().fit(classes)
        return self

    def transform(self, y):
        y = y.astype(str)
        unknown_mask = ~np.isin(y, self.classes_)
        if unknown_mask.any():
            # Opcional: advertencia
            print(f"[LabelEncoderWithUnknown] {unknown_mask.sum()} valores no vistos transformados a '__unknown__'")
        y_cleaned = np.where(unknown_mask, "__unknown__", y)
        return super().transform(y_cleaned)

    def fit_transform(self, y):
        return self.fit(y).transform(y)
