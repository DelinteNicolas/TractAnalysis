import os
import sys
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram
from dipy.tracking import utils


def register_atlas_to_subj(fa_path: str, atlas_path: str, mni_fa_path: str,
                           output_path: str):

    map_desikan_to_fa = find_transform(atlas_path, mni_fa_path,
                                       only_affine=True)
    map_mni_to_subj = find_transform(mni_fa_path, fa_path)

    inter_path = output_path[:-7] + '_inter.nii.gz'

    apply_transform(atlas_path, map_desikan_to_fa, static_file=mni_fa_path,
                    output_path=inter_path, labels=True)

    apply_transform(inter_path, map_mni_to_subj, static_file=fa_path,
                    output_path=output_path, labels=True)


def connectivity_matrices(dwi_path: str, labels_path: str, streamlines_path: str, output_path: str):

    dwi_data = nib.load(dwi_path).get_fdata()
    labels = nib.load(labels_path).get_fdata()

    trk = load_tractogram(streamlines_path, 'same')
    trk.to_corner()
    streams_data = trk.streamlines

    affine = nib.load(dwi_path).affine

    unique_values_and_count = np.unique(labels, return_counts=True)
    left_labels = []
    right_labels = []
    middle_labels = []
    for value in unique_values_and_count[0]:
        if 2000 <= value < 3000 or 4000 <= value < 5000 or value == 5002 or 43 <= value <= 63:  # Right label values
            right_labels.append(value)
        elif 14 <= value <= 16 or value == 24 or value == 85 or 251 <= value <= 255:  # Middle label values
            middle_labels.append(value)
        else:  # Left label values
            left_labels.append(value)
    left_labels = np.array(left_labels)
    right_labels = np.array(right_labels)
    middle_labels = np.array(middle_labels)
    labels_sorted = np.concatenate((left_labels, middle_labels, right_labels), axis=None)  # Concatenate to have an array that contains in order the left labels, then the middle ones and the right ones

    new_labels_value = np.linspace(0, len(labels_sorted) - 1, len(labels_sorted))
    new_label_map = np.zeros([len(labels), len(labels[0]), len(labels[0][0])])
    for i in range(len(labels_sorted)):
        new_label_map += np.where(labels == labels_sorted[i], new_labels_value[i], 0)
    new_label_map = np.round(new_label_map).astype(int)

    M, grouping = utils.connectivity_matrix(streams_data, affine,
                                            new_label_map,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)

    np.fill_diagonal(M, 0)
    M = M.astype('float32')
    M = M / np.sum(M)
    M = M[1:, 1:]

    im = plt.imshow(np.log1p(M * 100000), interpolation='nearest')
    plt.colorbar(im)
    plt.savefig(output_path + '_connectivity_matrix.png')

    np.save(output_path + '_connectivity_matrix.npy', M)


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
        patient_list = json.load(open(root + 'subjects/subj_list.json', "r"))

    path_to_analysis_code = root.replace(
        root.split('/')[-2] + '/', '') + 'TractAnalysis/'
    path_to_core = path_to_analysis_code + 'src/core.py'

    for patient in patient_list:

        os.system('sbatch -J ' + patient + ' '
                  + path_to_analysis_code + 'slurm/submitIter.sh '
                  + patient + ' ' + path_to_core + ' ' + root)


if __name__ == '__main__':

    # patient = sys.argv[1]
    # root = sys.argv[2]

    # path_to_analysis_code = root.replace(
    #     root.split('/')[-2] + '/', '') + 'TractAnalysis/'

    # fa_path = root + 'subjects/' + patient + '/dMRI/microstructure/dti/' + patient + '_FA.nii.gz'
    # atlas_path = path_to_analysis_code + 'data/atlas_desikan_killiany.nii.gz'
    # mni_fa_path = path_to_analysis_code + 'data/FSL_HCP1065_FA_1mm.nii.gz'
    # labels_path = root + 'subjects/' + patient + '/masks/' + patient + '_labels.nii.gz'

    # register_atlas_to_subj(fa_path, atlas_path, mni_fa_path, labels_path)

    # dwi_path = root + 'subjects/' + patient + '/dMRI/preproc/' + patient + '_dmri_preproc.nii.gz'
    # streamlines_path = root + 'subjects/' + patient + '/dMRI/tractography/' + patient + '_tractogram.trk'
    # matrix_path = root + 'subjects/' + patient + '/dMRI/tractography/' + patient

    dwi_path = 'C:/Users/dausort/Downloads/sub01_E1_dmri_preproc.nii.gz'
    labels_path = 'C:/Users/dausort/Downloads/sub01_E1_labels.nii.gz'
    streamlines_path = 'C:/Users/dausort/Downloads/sub01_E1_tracto_25_250000_1.trk'
    matrix_path = 'C:/Users/dausort/Downloads/sub01_E1'

    connectivity_matrices(dwi_path, labels_path, streamlines_path, matrix_path)
