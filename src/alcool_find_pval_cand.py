# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:53:46 2023

@author: nicol
"""

import json
from utils import (print_views_from_study_folder, get_mean_connectivity,
                   check_labels, labels_matching, metrics_analysis,
                   graphs_analysis, dictionary_without_controls,
                   difference_temps, dictionary_controls)
from core import (slurm_iter, significance_level, get_edges_of_interest,
                  register_labels_to_atlas, get_mean_tracts_study)


if __name__ == '__main__':

    # path_json = 'D:/TractAnalysis/output_analysis/unravel_metric_analysis.json'
    path_json = 'D:/TractAnalysis/output_analysis/essai_metric_analysis.json'
    control_path = 'D:/TractAnalysis/output_analysis/control_list.json'

    with open(control_path, 'r') as read_file:
        control_list = json.load(read_file)

    list_temps = ['E1', 'E2', 'E3']
    list_temps_control = ['E1', 'E2']

    dataframe_without_control = dictionary_without_controls(
        path_json, control_list)
    df_without_control = difference_temps(
        dataframe_without_control, control_list, 'Mean', True)
    df_without_control = df_without_control[list_temps]

    # ------------------------------------
    from scipy.stats import ttest_ind

    df = df_without_control

    region_list = list(df.index.unique(2))
    metric_list = list(df.index.unique(3))

    for r in region_list:

        dfr = df.xs(r, level=2)

        for m in metric_list:

            dfm = dfr.xs(m, level=2)
            p = ttest_ind(dfm['E1'].to_numpy().flatten(),
                          dfm['E3'].to_numpy().flatten(),
                          nan_policy='omit')[1]
            if p < 0.1:
                print(r, m, p)
