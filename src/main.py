import json
from utils import (print_views_from_study_folder, check_labels, labels_matching, get_min_connectivity, difference_temps, dictionary_patients)
from core import (register_labels_to_atlas, slurm_iter, significance_level, significance_level_evolution, get_edges_of_interest,
                  metrics_analysis)


if __name__ == '__main__':

    # Parameters
    root = '/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/alcoholic_study/'

    path_to_analysis_code = (root.replace(root.split('/')[-2] + '/', '') + 'TractAnalysis/')
    output_analysis_path = path_to_analysis_code + 'output_analysis/'

    labels_path = path_to_analysis_code + 'data/atlas_desikan_killiany.nii.gz'
    mni_fa_path = path_to_analysis_code + 'data/FSL_HCP1065_FA_1mm.nii.gz'
    label_atlas_path = output_analysis_path + 'atlas_desikan_killiany_mni.nii.gz'

    subjects_list_path = root + 'subjects/subj_list.json'
    with open(subjects_list_path, 'r') as read_file:
        list_subjects = json.load(read_file)
    control_list_path = root + 'subjects/control_list.json'
    with open(control_list_path, 'r') as read_file:
        list_control = json.load(read_file)

    excel_path = path_to_analysis_code + 'data/FreeSurfer_labels.xlsx'
    connectivity_matrix_index_file = (root + 'subjects/' + 'sub01_E1'
                                      + '/dMRI/tractography/' + 'sub01_E1'
                                      + '_labels_connectivity_matrix_sift.txt')

    pval_file = output_analysis_path + 'pvals_E12_E13_E23.npy'
    pval_evolution_file = output_analysis_path + 'pvals_E12.npy'
    min_path = output_analysis_path + 'min_connectivity_matrix.npy'
    min_evolution_path = output_analysis_path + 'min_connectivity_matrix_evolution.npy'

    patient_list = list(set(list_subjects) - set(list_control))

    metric_name = ['FA', 'AD', 'RD', 'MD', 'noddi_fintra', 'noddi_fextra',
                   'noddi_fiso', 'noddi_odi', 'diamond_wFA', 'diamond_wMD',
                   'diamond_wRD', 'diamond_wAD', 'diamond_frac_csf',
                   'diamond_frac_ftot', 'mf_frac_ftot', 'mf_wfvf',
                   'mf_frac_csf', 'mf_fvf_tot']

    selected_edges_path = output_analysis_path + 'selected_edges.json'

    # if not os.path.exists(path_to_analysis_code + 'output_analysis/'):
    #     os.mkdir(path_to_analysis_code + 'output_analysis/')

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
    # unwanted = []
    # p_list = [p for p in list_subjects if p not in unwanted]
    # check_labels(p_list, root, output_analysis_path)
    # labels_matching(excel_path, connectivity_matrix_index_file)

    # print('Computing p-values of connectivity matrices')
    # significance_level(patient_list, root, output_analysis_path)
    # significance_level_evolution(patient_list, list_control, root, output_analysis_path)

    # print('Computing min connectivity')
    # get_min_connectivity(output_analysis_path, False)
    # get_min_connectivity(output_analysis_path, True)

    # print('Finding most relevant connectivity edges')
    # get_edges_of_interest(pval_file, output_path=output_analysis_path,
    #                       min_path=min_path)
    # get_edges_of_interest(pval_evolution_file, output_path=output_analysis_path,
    #                       min_path=min_evolution_path)

    # print('Launching jobs to extract tract of interest')
    # slurm_iter(root, 'extraction')

# =============================================================================
# Third section - Computing tract microstructure
# =============================================================================

    # print('Dictionary of the ROI analysis for the selected edges')
    # path_json = metrics_analysis(patient_list, root, output_analysis_path, metric_name, selected_edges_path)

    print('Estimating mean tract microstructure metrics')
    slurm_iter(root, 'estimation', patient_list=['Third_section'])

# =============================================================================
# Fourth section - View results (Local)
# =============================================================================
    # see
