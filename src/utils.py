import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def get_view_from_data(data_path: str):

    img = nib.load(data_path)

    view = get_acquisition_view(img.affine)

    return view


def get_views_from_data_folder(folder_path: str):

    view_list = []

    for filename in os.listdir(folder_path):
        if '.nii' in filename:
            view = get_view_from_data(folder_path + '/' + filename)
            view_list.append(filename + '_' + view)

    return view_list


def print_views_from_study_folder(folder_path: str):
    '''


    Parameters
    ----------
    folder_path : str
        Path to study folder containing data_x folders

    '''

    view_list_tot = []

    for filename in os.listdir(folder_path):
        if os.path.isdir(folder_path + filename) and 'data_' in filename:
            view_list = get_views_from_data_folder(folder_path + '/' + filename)
            view_list_tot += view_list

    json.dump(view_list_tot, open(folder_path + 'subjects/subj_view.json', 'w'))


def get_acquisition_view(affine) -> str:
    '''
    Returns the acquisition view corresponding to the affine.

    Parameters
    ----------
    affine : 2-D array of shape (4,4)
        Array containing the affine information.

    Returns
    -------
    str
        String of the acquisition view, either 'axial', 'sagittal', 'coronal' or
        'oblique'.

    '''

    affine = affine[:3, :3].copy()

    sum_whole_aff = np.sum(abs(affine))
    sum_diag_aff = np.sum(np.diag(abs(affine)))
    sum_extra_diag_aff = sum_whole_aff - sum_diag_aff

    a = affine[0, 0]
    b = affine[1, 1]
    c = affine[2, 2]
    d = affine[0, 2]
    e = affine[1, 0]
    f = affine[2, 1]
    g = affine[0, 1]
    h = affine[1, 2]
    i = affine[2, 0]

    if (a != 0 and b != 0 and c != 0 and sum_extra_diag_aff == 0):
        return "axial"
    elif (d != 0 and e != 0 and f != 0 and sum_diag_aff == 0):
        return "sagittal"
    elif (g != 0 and h != 0 and i != 0 and sum_diag_aff == 0):
        return "coronal"
    else:
        return "oblique"


def get_mean_connectivity(list_subjects: list, root: str, output_path: str):
    '''


    Parameters
    ----------
    list_subjects : list
        DESCRIPTION.
    root : str
        DESCRIPTION.
    output_path : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    with open(list_subjects, 'r') as read_file:
        list_subjects = json.load(read_file)

    if os.path.isfile(output_path + 'labels_connectivity_matrix.txt'):
        labels_found = True
        with open(output_path + 'labels_connectivity_matrix.txt', 'r') as f:
            area_sorted = [line.rstrip('\n') for line in f]
    else:
        labels_found = False

    list_connectivity = []

    for i in range(len(list_subjects)):

        path = (root + 'subjects/' + str(list_subjects[i])
                + '/dMRI/tractography/' + str(list_subjects[i])
                + '_connectivity_matrix.npy')
        try:
            matrix = np.load(path)
            list_connectivity.append(matrix)
        except FileNotFoundError:
            continue

    list_connectivity = np.stack(list_connectivity, axis=2)

    mean_connectivity = np.mean(list_connectivity, axis=2)
    min_connectivity = np.min(list_connectivity, axis=2)

    fig, ax = plt.subplots()
    ax.imshow(np.log1p(mean_connectivity * 100000), interpolation='nearest')
    if labels_found:
        ax.set_yticks(np.arange(len(area_sorted)))
        ax.set_yticklabels(area_sorted)

    plt.savefig(output_path + 'mean_connectivity_matrix.png')

    np.save(output_path + 'mean_connectivity_matrix.npy', mean_connectivity)
    np.save(output_path + 'min_connectivity_matrix.npy', min_connectivity)
