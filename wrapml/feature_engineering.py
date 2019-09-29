import pandas as pd
import numpy as np 
from sklearn.base import TransformerMixin

class MeanEncoder(TransformerMixin):
    def __init__(self, missing_value=-1):
        self.mean_map = {}
        self.missing_value = missing_value

    def fit(self, X, target):
        self.mean_map = pd.DataFrame({'value': X, 'target': target}).groupby('value')['target'].mean().to_dict()

    def transform(self, X):
        X_cp = X.copy()
        X_cp[~X.isin(self.mean_map.keys())] = self.missing_value
        X_cp = X_cp.map(self.mean_map)
        return X_cp


class FrequencyEncoder(TransformerMixin):
    def __init__(self, missing_value=-1):
        self.count_map = {}
        self.missing_value = missing_value
        
    def fit(self, X):
        self.count_map = X.value_counts().to_dict()
        
    def transform(self, X):
        X_cp = X.copy()
        X_cp[~X.isin(self.count_map.keys())] = self.missing_value
        X_cp = X_cp.map(self.count_map)
        return X_cp
    
class LabelEncoderExt(TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, data_list):
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_
        return self

    def transform(self, data_list):
        data_list_cpy = data_list.copy()
        unknown_index = ~(data_list.isin(self.classes_))
        data_list_cpy[unknown_index] = 'Unknown'
        return self.label_encoder.transform(data_list_cpy)


    
class DataFrameEncoder:
    def __init__(self, encoder, columns, col_prefix=''):
        self.encoder = encoder
        self.columns = columns
        self.col_prefix = col_prefix
        self.encoder_map = {}
    
    def fit(self, df, target=None):
        for col in self.columns:
            _en = self.encoder()
            if target is not None:
                _en.fit(df[col], target)
            else:
                _en.fit(df[col])
            self.encoder_map[col] = _en
            
    def transform(self, df):
        for col in self.columns:
            if col not in self.encoder_map.keys():
                raise ValueError(f"Unexpected column {col} in transform")
            _en = self.encoder_map[col]
            encoded_col = _en.transform(df[col])
            df[self.col_prefix + col] = encoded_col
        return df
    
    def fit_transform(self, df, target=None):
        self.fit(df, target)
        return self.transform(df)

class DataFrameColumnCombiner:
    def __init__(self, combine_columns, copy=False):
        self.combine_columns = combine_columns
        self.copy = copy

    def combine_columns(self, df, cols):
        return df[cols].apply(lambda x: '_'.join([str(s) for s in x]), axis=1)

    def transform(df):
        if copy: dft = df.copy()
        else: dft = df
        if type(self.combine_columns) is list:
            for cols in self.combine_columns:
                col_name = '_'.join(cols)
                dft[col_name] = self.combine_columns(dft, cols)
        elif type(self.combite_columns) is dict:
            for col_name, cols in self.combine_columns.items():
                dft[col_name] = self.combine_columns(dft, cols)
        return dft

def get_high_correlation_cols(df, corrThresh=0.9):
    numeric_cols = df._get_numeric_data().columns
    corr_matrix = df.loc[:, numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corrThresh)]
    return to_drop

def plot_feature_importances(fe, cols):
    fe = pd.DataFrame(fe, index=cols)
    if fe.shape[1] > 1:
        fe = fe.apply(sum, axis=1)
    else:
        fe = fe[0]
    fe.sort_values(ascending=False)[:20].plot(kind='bar')

def replace_na(data, numeric_replace=-1, categorical_replace='missing', cat_features=[]):
    numeric_cols = data._get_numeric_data().columns
    categorical_cols = list(set(list(set(data.columns) - set(numeric_cols)) + cat_features))
    categorical_cols = [col for col in categorical_cols if col in data.columns]
    if numeric_replace is not None:
        data[numeric_cols] = data[numeric_cols].fillna(numeric_replace)
    data[categorical_cols] = data[categorical_cols].fillna(categorical_replace)
    return data

def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan) 

def get_null_columns(df, threshold=0.9):
    return list(df.columns[(df.isnull().sum() / df.shape[0]) > 0.9])    

def get_big_top_value_columns(df, threshold=0.9):
    return [col for col in df.columns if 
            df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

def get_non_numeric_columns(df):
    num_cols = df._get_numberic_data().columns
    return [col for col in df.columns if col not in num_cols]



