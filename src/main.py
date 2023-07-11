from utils import print_views_from_study_folder
from core import slurm_iter


if __name__ == '__main__':

    root = '/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/alcoholic_study/'

    # Checking views
    print_views_from_study_folder(root)

    slurm_iter(root, patient_list=['sub01_E1'])
