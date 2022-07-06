"""
module: datset_calling is to import dataset
authored: Phuong V. Nguyen
dated: August 31th, 2021
"""
from IPython.display import display, HTML, clear_output, update_display
import pandas as pd
import os.path
import logging


def load_data(name_dataset='Iris',
             verbose=True,
             address="https://raw.githubusercontent.com/thangnv11/mlop_beta/main/datasets/",):
    """
    This function is to load dataset from the git repository
    :param name_dataset: name of dataset (str)
    :param verbose:
    :param address: url of dataset in the git repo
    :return data: loaded dataset (pandas.DataFrame)
    """
    extension = ".csv"
    filename = str(name_dataset) + extension
    full_address = address + filename
    logging.info('data loading\n...')
    data = pd.read_csv(full_address)
    if verbose:
        display(data.head())
    return data
