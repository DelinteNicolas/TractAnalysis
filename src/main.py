import json
from utils import (print_views_from_study_folder, get_mean_connectivity,
                   check_labels, labels_matching)
from core import (slurm_iter, significance_level, get_edges_of_interest,
                  register_labels_to_atlas, get_mean_tracts_study)


if __name__ == '__main__':

    # Parameters
    root = '/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/alcoholic_study/'

    path_to_analysis_code = (root.replace(root.split('/')[-2] + '/', '')
                             + 'TractAnalysis/')
    subjects_list = root + 'subjects/subj_list.json'
    output_analysis_path = path_to_analysis_code + 'output_analysis/'
    pval_file = output_analysis_path + 'pvals_E12_E13_E23.npy'
    min_path = output_analysis_path + 'min_connectivity_matrix.npy'
    labels_path = path_to_analysis_code + 'data/atlas_desikan_killiany.nii.gz'
    mni_fa_path = path_to_analysis_code + 'data/FSL_HCP1065_FA_1mm.nii.gz'
    label_atlas_path = output_analysis_path + 'atlas_desikan_killiany_mni.nii.gz'
    freeSurfer_labels = path_to_analysis_code + 'data/FreeSurfer_labels.xlsx'
    selected_edges_path = output_analysis_path+'selected_edges.json'

    with open(subjects_list, 'r') as read_file:
        list_subjects = json.load(read_file)

    connectivity_matrix_index_file = (root + 'subjects/' + 'sub01_E1'
                                      + '/dMRI/tractography/' + 'sub01_E1'
                                      + '_labels_connectivity_matrix_sift.txt')

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
    # significance_level(subjects_list, root, output_analysis_path)

    # print('Computing mean connectivity')
    # get_mean_connectivity(subjects_list, root, output_analysis_path)

    # print('Finding most relevant connectivity edges')
    # edge = get_edges_of_interest(pval_file, output_path=output_analysis_path,
    #                              min_path=min_path)

    # print('Launching jobs to extract tract of interest')
    # slurm_iter(root, 'extraction', patient_list=['sub01_E1'])

# =============================================================================
# Third section - Computing tract microstructure
# =============================================================================

    print('Estimating mean tract microscture metrics')
    get_mean_tracts_study(root, selected_edges_path, output_analysis_path)
