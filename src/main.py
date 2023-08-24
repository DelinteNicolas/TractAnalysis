import json
from utils import (print_views_from_study_folder, get_mean_connectivity,
                   check_labels, labels_matching, metrics_analysis)
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
    selected_edges_path = output_analysis_path + 'selected_edges.json'

    with open(subjects_list, 'r') as read_file:
        list_subjects = json.load(read_file)

    connectivity_matrix_index_file = (root + 'subjects/' + 'sub01_E1'
                                      + '/dMRI/tractography/' + 'sub01_E1'
                                      + '_labels_connectivity_matrix_sift.txt')

    patient_list = ["sub01_E1", "sub01_E2", "sub02_E1", "sub02_E2", "sub02_E3",
                    "sub03_E2", "sub03_E3", "sub04_E1", "sub04_E2", "sub05_E1",
                    "sub05_E2", "sub06_E2", "sub07_E2", "sub08_E1", "sub08_E2",
                    "sub09_E1", "sub09_E2", "sub09_E3", "sub10_E1", "sub11_E1",
                    "sub11_E2", "sub12_E1", "sub12_E2", "sub13_E1", "sub13_E2",
                    "sub14_E1", "sub14_E2", "sub15_E1", "sub15_E2", "sub16_E1",
                    "sub17_E1", "sub17_E2", "sub17_E3", "sub18_E1", "sub18_E2",
                    "sub19_E1", "sub19_E2", "sub20_E1", "sub20_E2", "sub20_E3",
                    "sub21_E1", "sub21_E2", "sub22_E1", "sub22_E2", "sub22_E3",
                    "sub23_E1", "sub24_E1", "sub24_E2", "sub25_E2", "sub26_E1",
                    "sub26_E2", "sub27_E1", "sub27_E2", "sub27_E3", "sub28_E1",
                    "sub28_E2", "sub29_E1", "sub29_E2", "sub30_E1", "sub30_E2",
                    "sub30_E3", "sub31_E1", "sub31_E2", "sub32_E1", "sub32_E2",
                    "sub33_E1", "sub33_E2", "sub34_E1", "sub34_E2", "sub34_E3",
                    "sub35_E1", "sub35_E2", "sub36_E1", "sub36_E2", "sub36_E3",
                    "sub37_E1", "sub37_E2", "sub38_E1", "sub39_E1", "sub39_E2",
                    "sub39_E3", "sub40_E1", "sub40_E2", "sub41_E1", "sub41_E2",
                    "sub42_E1", "sub42_E2", "sub43_E1", "sub43_E2", "sub44_E1",
                    "sub45_E1", "sub45_E2", "sub45_E3", "sub46_E1", "sub46_E2",
                    "sub47_E1", "sub48_E1", "sub48_E2", "sub50_E1", "sub50_E2",
                    "sub51_E1", "sub51_E2", "sub52_E1", "sub52_E2", "sub53_E1",
                    "sub53_E2", "sub54_E1", "sub54_E2", "sub55_E1", "sub55_E2",
                    "sub56_E1", "sub57_E1", "sub57_E2", "sub58_E1", "sub58_E2",
                    "sub59_E1", "sub59_E2", "sub60_E1", "sub60_E2", "sub61_E1",
                    "sub61_E2", "sub62_E1", "sub62_E2", "sub63_E1", "sub63_E2",
                    "sub64_E1", "sub64_E2", "sub65_E1", "sub65_E2", "sub66_E1",
                    "sub66_E2", "sub67_E2", "sub68_E1", "sub68_E2",
                    "sub69_E1", "sub69_E2", "sub70_E1", "sub70_E2", "sub71_E1",
                    "sub71_E2", "sub72_E1", "sub72_E2", "sub73_E1", "sub73_E2"]

    metric_name = ['FA', 'AD', 'RD', 'MD', 'noddi_fintra', 'noddi_fextra', 'noddi_fiso', 'noddi_odi',
                   'diamond_wFA', 'diamond_wMD', 'diamond_wRD', 'diamond_wAD',
                   'diamond_fractions_csf', 'diamond_fractions_ftot', 'mf_frac_ftot',
                   'mf_wfvf', 'mf_frac_csf', 'mf_fvf_tot']

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

    metrics_analysis(patient_list, root, output_analysis_path, metric_name, selected_edges_path)

    # print('Launching jobs to extract tract of interest')
    # slurm_iter(root, 'extraction')  # , patient_list=['sub01_E1'])

# =============================================================================
# Third section - Computing tract microstructure
# =============================================================================

    # print('Estimating mean tract microscture metrics')
    # slurm_iter(root, 'estimation', patient_list=['Third_section'])
