"""
This module is to do the Exploratory Data Analysis (EDA)
Authored: Phuong V. Nguyen
Dated: August 31th, 2021
"""
import pandas as pd
import numpy as np
from IPython.display import display, HTML, clear_output, update_display


def discovery(data: pd.DataFrame,
            feature: [],
            label: str,
            profile=False,
            sweetplot=False,
            autoplot=False,
            tale=True
            ):
    """
    This function is do EAD both descriptive statistics and visualization
    :param data:
    :param feature: list of features
    :param label: target variable
    :param profile: using pandas-profiling
    :param sweetplot: using sweetvis
    :param autoplot: using autovis
    :param dtale: using D-tale
    :return state: result of EDA
    """
    if profile:
        import pandas_profiling
        pf = pandas_profiling.ProfileReport(data)
        display(pf)
        state = 'EDA WITH PROFILE IS DONE'
    elif sweetplot:
        import sweetviz
        my_report = sweetviz.analyze([data, 'Train'],
                                     target_feat=label)
        my_report.show_html('FinalReport.html')
        state = 'EDA WITH SWEETPLOT IS DONE'
    else:
        if autoplot:
            from autoviz.AutoViz_Class import AutoViz_Class
            AV = AutoViz_Class()
            df = AV.AutoViz(data)
            print('Sorry, this task is under construction')
        elif tale:
            import dtale
            d = dtale.show(data)
            d.open_browser()
            state = 'EDA WITH TALE IS DONE'
    return state

