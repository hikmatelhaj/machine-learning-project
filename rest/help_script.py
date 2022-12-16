from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.
    """
    def __init__(self):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()
        self.feature_labels=[]

    def fit(self, X:pd.DataFrame, y=None):
        for i in range(X.shape[1]): # X can be of multiple columns
            mlb = MultiLabelBinarizer()
            mlb.fit(X.iloc[:,i])
            self.mlbs.append(mlb)
            self.classes_.append(mlb.classes_)
            self.feature_labels.extend(mlb.classes_)
            self.n_columns += 1
        return self

    def transform(self, X:pd.DataFrame):
        if self.n_columns == 0:
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:,i]))

        result = np.concatenate(result, axis=1)
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_labels
        # cats = self.classes_
        # if input_features is None:
        #     input_features = ['x%d' % i for i in range(len(cats))]
        #     print(input_features)
        # elif len(input_features) != len(self.categories_):
        #     raise ValueError(
        #         "input_features should have length equal to number of "
        #         "features ({}), got {}".format(len(self.categories_),
        #                                        len(input_features)))

        # feature_names = [f"{input_features[i]}_{cats[i]}" for i in range(len(cats))]
        # return np.array(feature_names, dtype=object)