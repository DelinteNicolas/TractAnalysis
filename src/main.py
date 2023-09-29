import json
import os
from utils import (print_views_from_study_folder, get_mean_connectivity,
                   check_labels, labels_matching, metrics_analysis,
                   graphs_analysis, dictionary_without_controls,
                   difference_temps)
from core import (slurm_iter, significance_level, get_edges_of_interest,
                  register_labels_to_atlas, get_mean_tracts_study)


if __name__ == '__main__':

    # Parameters
    root = '/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/alcoholic_study/'

    path_to_analysis_code = (root.replace(root.split('/')[-2] + '/', '')
                             + 'TractAnalysis/')

    subjects_list_path = root + 'subjects/subj_list.json'
    control_list_path = root + 'subjects/control_list.json'

    if not os.path.exists(path_to_analysis_code + 'output_analysis/'):
        os.mkdir(path_to_analysis_code + 'output_analysis/')

    output_analysis_path = path_to_analysis_code + 'output_analysis/'
    pval_file = output_analysis_path + 'pvals_E12_E13_E23.npy'
    min_path = output_analysis_path + 'min_connectivity_matrix.npy'
    labels_path = path_to_analysis_code + 'data/atlas_desikan_killiany.nii.gz'
    mni_fa_path = path_to_analysis_code + 'data/FSL_HCP1065_FA_1mm.nii.gz'
    label_atlas_path = output_analysis_path + 'atlas_desikan_killiany_mni.nii.gz'
    freeSurfer_labels = path_to_analysis_code + 'data/FreeSurfer_labels.xlsx'
    selected_edges_path = output_analysis_path + 'selected_edges.json'

    with open(subjects_list_path, 'r') as read_file:
        list_subjects = json.load(read_file)

    with open(control_list_path, 'r') as read_file:
        list_control = json.load(read_file)

    patient_list = list(set(list_subjects) - set(list_control))

    connectivity_matrix_index_file = (root + 'subjects/' + 'sub01_E1'
                                      + '/dMRI/tractography/' + 'sub01_E1'
                                      + '_labels_connectivity_matrix_sift.txt')

    metric_name = ['FA', 'AD', 'RD', 'MD', 'noddi_fintra', 'noddi_fextra',
                   'noddi_fiso', 'noddi_odi', 'diamond_wFA', 'diamond_wMD',
                   'diamond_wRD', 'diamond_wAD', 'diamond_fractions_csf',
                   'diamond_fractions_ftot', 'mf_frac_ftot', 'mf_wfvf',
                   'mf_frac_csf', 'mf_fvf_tot']

# =============================================================================
# First section - Connectivity
# =============================================================================

    # print('Checking view orientations for all patients')
    # print_views_from_study_folder(root)

    # print('Sending labels to mni space')
    # register_labels_to_atlas(labels_path, mni_fa_path, label_atlas_path)

    # print('Launching jobs to compute connectivity matrices')
    # slurm_iter(root, 'connectivity')

# =============================================================================
# Second section - Tract extraction
# =============================================================================

    # print('Verification of labels')
    # # unwanted = ['sub13_E1', 'sub56_E1', 'sub304_E1']
    # unwanted = []
    # p_list = [p for p in list_subjects if p not in unwanted]
    # check_labels(p_list, root, output_analysis_path)

    # print('Indexing new labels in excel file')
    # labels_matching(freeSurfer_labels, connectivity_matrix_index_file)

    # print('Computing p-values of connectivity matrices')
    # significance_level(subjects_list_path, root, output_analysis_path)

    # print('Computing mean connectivity')
    # get_mean_connectivity(subjects_list_path, root, output_analysis_path)

    # print('Finding most relevant connectivity edges')
    # get_edges_of_interest(pval_file, output_path=output_analysis_path,
    #                       min_path=min_path)

    # print('Launching jobs to extract tract of interest')
    # slurm_iter(root, 'extraction')  # , patient_list=['sub01_E1'])

# =============================================================================
# Third section - Computing tract microstructure
# =============================================================================

    # print('Dictionary of the ROI analysis for the selected edges')
    # path_json = metrics_analysis(patient_list, root, output_analysis_path,
    #                              metric_name, selected_edges_path)

    # print('Estimating mean tract microstructure metrics')
    # slurm_iter(root, 'estimation', patient_list=['Third_section'])

# =============================================================================
# Fourth section - View results (Local)
# =============================================================================

    # path_json = 'D:/TractAnalysis/output_analysis/unravel_metric_analysis.json'
    # control_path = 'D:/TractAnalysis/output_analysis/control_list.json'

    # print('Graph analysis for a specific dictionary, a specific region and a '
    #       + 'specific metric')

    # with open(control_path, 'r') as read_file:
    #     control_list = json.load(read_file)

    # list_temps = ['E1', 'E2', 'E3']
    # list_temps_control = ['E1', 'E2']

    # dataframe_without_control = dictionary_without_controls(
    #     path_json, control_list)
    # df_without_control = difference_temps(
    #     dataframe_without_control, control_list, 'Mean', True)
    # df_without_control = df_without_control[list_temps]

    # a = graphs_analysis(df_without_control, '[43, 29]', 'Mean', 'stream_count',
    #                     list_temps)
