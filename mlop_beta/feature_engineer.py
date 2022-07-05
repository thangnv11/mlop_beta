"""
This module is to do feature engineer.
Authored: Phuong V. Nguyen
Dated: August 21th, 2021
"""
import pandas as pd
import numpy as np
import ipywidgets as wg
from IPython.display import display
from ipywidgets import Layout
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute._base import _BaseImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


def basic_feat_engineer(data, features_to_drop):
    """
    this function is to do some basic aspects of feature engineer
    :param data: input data
    :param features_to_drop: list of features were asked to drop
    :return df: cleaned data
    """

    df = data.copy()

    # also make sure that all the column names are string
    df.columns = [str(i) for i in df.columns]

    # drop any columns that were asked to drop
    df.drop(columns=features_to_drop, errors="ignore", inplace=True)

    # if there are inf or -inf then replace them with NaN
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)

    # if data type is bool or pandas Categorical , convert to categorical
    for i in df.select_dtypes(include=["bool", "category"]).columns:
        df[i] = df[i].astype("object")

    # wiith csv , if we have any null in  a colum that was int , panda will read it as float.
    # so first we need to convert any such floats that have NaN and unique values are
    # lower than 20
    for i in df.select_dtypes(include=["float64"]).columns:
        df[i] = df[i].astype("float32")
        # count how many Nas are there
        na_count = sum(data[i].isnull())
        # count how many digits are there that have decimiles
        count_float = np.nansum(
            [False if r.is_integer() else True for r in data[i]]
        )
        # total decimiels digits
        count_float = (
                count_float - na_count
        )  # reducing it because we know NaN is counted as a float digit
        # now if there isnt any float digit , & unique levales are less than
        # 20 and there are Na's then convert it to object
        if (count_float == 0) & (df[i].nunique() <= 20) & (na_count > 0):
            df[i] = df[i].astype("object")

    return df


def data_transformer(x: pd.DataFrame,
                     rescale=False,
                     standard=False):
    """
    This function is to transform data by rescaling method
    :param x: input data
    :param rescale: condition if using the rescaling method
    :param standard: condition if using standardizing method
    :return df_rescaledX: data after transforming
    """
    if rescale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(x)
        df_rescaledX = pd.DataFrame(rescaledX,
                                    index=x.index, columns=x.columns)
    elif standard:
        print('This function is underconstruction')
    return df_rescaledX


def one_hot_encoder(df: pd.DataFrame,
                   encode_var: []):
    """
    This function is to do one_hot_encoding
    :param df: input dataframe
    :param encode_var: list of variables is asked to encode
    :return one_hot_encoded_catDF: data after one-hot-encoding
    """
    if df.dtypes[0] != 'object':
        catDF = df.astype('object')
    else:
        catDF = df
    one_hot_encoded_catDF = pd.get_dummies(catDF, columns = encode_var,
                                           dummy_na=True, drop_first=True)
    return one_hot_encoded_catDF


def ft_engineer(data: pd.DataFrame,
                label: str,
                cat_var: [],
                features_to_drop: [],
                transform=False,
                onehot=False):
    """
    This function is to do feature engineering in pipeline
    :param data:
    :param label:
    :param cat_var:
    :param features_to_drop:
    :param transform:
    :param onehot:
    :return x_train, x_test, y_train, y_test:
    """
    df = basic_feat_engineer(data, features_to_drop)
    x = df.drop(label, axis=1)
    y = df[label]
    if transform:
        numericalX = x.drop(cat_var, axis=1)
        transX = data_transformer(numericalX, rescale = True)
    else:
        print('no numerical variables')
        transX = x.drop(cat_var, axis=1)
    transX[cat_var] = x[cat_var]
    if onehot:
        encoded_transX = one_hot_encoder(transX, cat_var)
    else:
        encoded_transX = transX
    x_train, x_test, y_train, y_test = train_test_split(encoded_transX, y,
                                                        test_size=0.30, random_state=42)
    print('X train: {}'.format(x_train.shape))
    print('y train: {}'.format(y_train.shape))
    print('X test: {}'.format(x_test.shape))
    print('y test: {}'.format(y_test.shape))
    return x_train, x_test, y_train, y_test
