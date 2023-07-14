from utils import print_views_from_study_folder
from core import slurm_iter, significance_level, get_edges_of_interest


if __name__ == '__main__':

    root = '/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/alcoholic_study/'

    path_to_analysis_code = root.replace(
        root.split('/')[-2] + '/', '') + 'TractAnalysis/'

    # Checking views
    # print('Checking view orientations for all patients')
    # print_views_from_study_folder(root)

    # print('Launching jobs to compute connectivity matrices')
    # slurm_iter(root, 'connectivity', patient_list=['sub01_E1'])

    # Pvals
    subjects_list = root + 'subjects/subj_list.json'
    output_path = path_to_analysis_code + 'output_analysis/'

    # print('Computing p-values of connectivity matrices')
    # significance_level(subjects_list, root, output_path)

    pval_file = path_to_analysis_code + 'output_analysis/_pvals_E12_E13_E23.npy'

    print('Finding most relevant connectivity edges')
    edge = get_edges_of_interest(pval_file, output_path=output_path)

    print('Launching jobs to extract tract of interest')
    slurm_iter(root, 'extraction', patient_list=['sub01_E1'])
