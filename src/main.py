from utils import print_views_from_study_folder, get_mean_connectivity, check_labels
from core import slurm_iter, significance_level, get_edges_of_interest


if __name__ == '__main__':

    root = '/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/alcoholic_study/'

    path_to_analysis_code = root.replace(
        root.split('/')[-2] + '/', '') + 'TractAnalysis/'
    subjects_list = root + 'subjects/subj_list.json'
    output_path = path_to_analysis_code + 'output_analysis/'
    pval_file = path_to_analysis_code + 'output_analysis/_pvals_E12_E13_E23.npy'
    min_path = path_to_analysis_code + 'output_analys/min_connectivity_matrix.npy'

# =============================================================================
# First section - Connectivity
# =============================================================================

    # print('Checking view orientations for all patients')
    # print_views_from_study_folder(root)

    # with open(subjects_list, 'r') as read_file:
    #    list_subjects = json.load(read_file)

    # print('Launching jobs to compute connectivity matrices')
    # slurm_iter(root, 'connectivity', patient_list=['sub01_E1', 'sub01_E2'])

# =============================================================================
# Second section - Tract extraction
# =============================================================================

    print('Verification of labels')
    unwanted = ['sub13_E1', 'sub56_E1', 'sub304_E1']
    p_list = [p for p in subjects_list if p not in unwanted]
    check_labels(p_list, root, output_path)

    print('Computing p-values of connectivity matrices')
    significance_level(subjects_list, root, output_path)

    print('Get mean connectivity')
    get_mean_connectivity(subjects_list, root, output_path)

    print('Finding most relevant connectivity edges')
    edge = get_edges_of_interest(pval_file, output_path=output_path,
                                 min_path=min_path)

    print('Launching jobs to extract tract of interest')
    slurm_iter(root, 'extraction', patient_list=['sub01_E1'])
