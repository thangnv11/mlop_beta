"""
module: datset_calling is to import dataset
authored: Phuong V. Nguyen
dated: August 31th, 2021
"""
import pandas as pd
import os.path
from IPython.display import display, HTML, clear_output, update_display

def load_data(name_dataset='index',
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
    data = pd.read_csv(full_address)
    if verbose:
        display(data.head())
    return data
