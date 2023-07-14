import os
import sys
import json
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from regis.core import find_transform, apply_transform
from dipy.io.streamline import load_tractogram
from dipy.tracking import utils
from scipy.stats import ttest_ind


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


def connectivity_matrices(dwi_path: str, labels_path: str, streamlines_path: str, output_path: str, freeSurfer_labels: str):

    dwi_data = nib.load(dwi_path).get_fdata()
    labels = nib.load(labels_path).get_fdata()

    trk = load_tractogram(streamlines_path, 'same')
    trk.to_corner()
    streams_data = trk.streamlines

    affine = nib.load(dwi_path).affine

    data = pd.read_excel(freeSurfer_labels)
    df = pd.DataFrame(data)

    values, values_counts = np.unique(labels, return_counts=True)

    right_labels = []
    right_area = []
    left_labels = []
    left_area = []
    middle_labels = []
    middle_area = []

    for i in range(len(values)):
        for j in range(len(df['Index'])):
            if (values[i] == df['Index'][j]):
                if 'right' in str(df['Area'][j]):
                    right_labels.append(values[i])
                    right_area.append(df['Area'][j])
                elif 'left' in str(df['Area'][j]):
                    left_labels.append(values[i])
                    left_area.append(df['Area'][j])
                else:
                    middle_labels.append(values[i])
                    middle_area.append(df['Area'][j])

    labels_sorted = np.append(np.append(right_labels, middle_labels), left_labels)
    a = np.argwhere(labels_sorted == 0)

    labels_sorted = np.delete(labels_sorted, 50)
    area_sorted = np.append(np.append(right_area, middle_area), left_area)
    area_sorted = np.delete(area_sorted, 50)

    new_labels_value = np.linspace(0, len(labels_sorted) - 1, len(labels_sorted))
    new_label_map = np.zeros([len(labels), len(labels[0]), len(labels[0][0])])
    for i in range(len(labels_sorted)):
        new_label_map += np.where(labels == labels_sorted[i], new_labels_value[i], 0)
    new_label_map = new_label_map.astype('int64')

    M, grouping = utils.connectivity_matrix(streams_data, affine,
                                            new_label_map,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)

    np.fill_diagonal(M, 0)
    M = M.astype('float32')
    M = M / np.sum(M)

    fig, ax = plt.subplots()
    ax.imshow(np.log1p(M * 100000), interpolation='nearest')
    ax.set_yticks(np.arange(len(area_sorted)))
    ax.set_yticklabels(area_sorted)

    plt.savefig(output_path + '_connectivity_matrix.png')

    np.save(output_path + '_connectivity_matrix.npy', M)

    return new_label_map

def significance_level(list_subject: str, root: str, output_path: str):

    with open(list_subject, 'r') as read_file:
        list_subject = json.load(read_file)

    list_E1 = []
    list_E2 = []
    list_E3 = []

    for i in range(len(list_subject)):

        path = root + 'subjects/' + str(list_subject[i]) + '/dMRI/tractography/' + str(list_subject[i]) + '_connectivity_matrix.npy'
        try:
            matrix = np.load(path)
        except FileNotFoundError:
            continue

        if 'E1' in str(list_subject[i]):
            list_E1.append(matrix)
        elif 'E2' in str(list_subject[i]):
            list_E2.append(matrix)
        else:
            list_E3.append(matrix)

    list_E1 = np.stack(list_E1, axis=2)
    list_E2 = np.stack(list_E2, axis=2)
    list_E3 = np.stack(list_E3, axis=2)

    # On part du principe que les entrées des matrices sont les mêmes mais à vérif
    pval_E12 = np.zeros((list_E1.shape[0], list_E1.shape[1]))
    pval_E13 = np.zeros((list_E1.shape[0], list_E1.shape[1]))
    pval_E23 = np.zeros((list_E1.shape[0], list_E1.shape[1]))

    for i in range(list_E1.shape[0]):
        for j in range(list_E1.shape[1]):
            _, pval_12 = ttest_ind(list_E1[i, j, :], list_E2[i, j, :], alternative='two-sided')
            pval_E12[i, j] = pval_12

            _, pval_13 = ttest_ind(list_E1[i, j, :], list_E3[i, j, :], alternative='two-sided')
            pval_E13[i, j] = pval_13

            _, pval_23 = ttest_ind(list_E2[i, j, :], list_E3[i, j, :], alternative='two-sided')
            pval_E23[i, j] = pval_23

    pval_all = []
    pval_all = pval_all.append(pval_E12)
    pval_all = pval_all.append(pval_E13)  # .append(pval_E23)
    pval_all = np.stack(pval_all, axis=2)

    np.save(output_path + '_pvals_E12_E13_E23.npy', pval_all)


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

    patient = sys.argv[1]
    root = sys.argv[2]

    path_to_analysis_code = root.replace(
        root.split('/')[-2] + '/', '') + 'TractAnalysis/'

    fa_path = root + 'subjects/' + patient + '/dMRI/microstructure/dti/' + patient + '_FA.nii.gz'
    atlas_path = path_to_analysis_code + 'data/atlas_desikan_killiany.nii.gz'
    mni_fa_path = path_to_analysis_code + 'data/FSL_HCP1065_FA_1mm.nii.gz'
    labels_path = root + 'subjects/' + patient + '/masks/' + patient + '_labels.nii.gz'

    register_atlas_to_subj(fa_path, atlas_path, mni_fa_path, labels_path)

    dwi_path = root + 'subjects/' + patient + '/dMRI/preproc/' + patient + '_dmri_preproc.nii.gz'
    streamlines_path = root + 'subjects/' + patient + '/dMRI/tractography/' + patient + '_tractogram.trk'
    matrix_path = root + 'subjects/' + patient + '/dMRI/tractography/' + patient

    subjects_list = root + 'subjects/subj_list.json'
    freeSurfer_labels = path_to_analysis_code + 'data/FreeSurfer_labels.xlsx'
    output_path = path_to_analysis_code + 'output_analysis/'

    new_label_map = connectivity_matrices(dwi_path, labels_path, streamlines_path, matrix_path, freeSurfer_labels)
