import os
import sys
import json
from regis.core import find_transform, apply_transform


def register_atlas_to_subj(fa_path: str, atlas_path: str, mni_fa_path: str,
                           output_path: str):

    map_desikan_to_fa = find_transform(atlas_path, mni_fa_path,
                                       only_affine=True)
    map_mni_to_subj = find_transform(mni_fa_path, fa_path)

    inter_path = output_path[:-7]+'_inter.nii.gz'

    apply_transform(atlas_path, map_desikan_to_fa, static_file=mni_fa_path,
                    output_path=inter_path, labels=True)

    apply_transform(inter_path, map_mni_to_subj, static_file=fa_path,
                    output_path=output_path, labels=True)


def slurm_iter(root: str, patient_list: list = []):
    '''


    Parameters
    ----------
    root : str
        Path to study
    patient_list : list, optional
        If not specified, runs on all patients in the study. The default is [].

    Returns
    -------
    None.

    '''

    if len(patient_list) == 0:
        patient_list = json.load(open(root+'subjects/subj_list.json', "r"))

    path_to_analysis_code = root.replace(
        root.split('/')[-2]+'/', '')+'TractAnalysis/'
    path_to_core = path_to_analysis_code+'src/core.py'

    for patient in patient_list:

        os.system('sbatch -J '+patient+' '
                  + path_to_analysis_code+'slurm/submitIter.sh '
                  + patient+' '+path_to_core+' '+root)


if __name__ == '__main__':

    patient = sys.argv[1]
    root = sys.argv[2]

    path_to_analysis_code = root.replace(
        root.split('/')[-2]+'/', '')+'TractAnalysis/'

    fa_path = root+'subjects/'+patient+'/dMRI/microstructure/dti/'+patient+'_FA.nii.gz'
    atlas_path = path_to_analysis_code+'data/atlas_desikan_killiany.nii.gz'
    mni_fa_path = path_to_analysis_code+'data/FSL_HCP1065_FA_1mm.nii.gz'
    output_path = root+'subjects/'+patient+'/masks/'+patient+'_labels.nii.gz'

    register_atlas_to_subj(fa_path, atlas_path, mni_fa_path, output_path)
