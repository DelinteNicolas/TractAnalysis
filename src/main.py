from utils import print_views_from_study_folder
from core import slurm_iter
from core import significance_level


if __name__ == '__main__':

    root = '/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/alcoholic_study/'

    path_to_analysis_code = root.replace(
        root.split('/')[-2] + '/', '') + 'TractAnalysis/'

    # Checking views
    print_views_from_study_folder(root)

    slurm_iter(root, patient_list=['sub01_E1'])

    # Pvals
    subjects_list = root + 'subjects/subj_list.json'
    output_path = path_to_analysis_code + 'output_analysis/'

    significance_level(subjects_list, root, output_path)
