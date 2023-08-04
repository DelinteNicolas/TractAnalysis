import os
import json
import string
import pandas as pd
import numpy as np
import xlsxwriter
import nibabel as nib
from unravel.utils import tract_to_ROI
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


def get_mean_connectivity(list_subjects: str, root: str, output_path: str):
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
                + '_connectivity_matrix_sift.npy')
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


def check_labels(list_subjects: str, root: str, output_path: str):

    general_list = []
    check_failed = False

    # with open(list_subjects, 'r') as read_file:
    #     list_subjects = json.load(read_file)

    for i in range(len(list_subjects)):

        with open(root + "subjects/" + str(list_subjects[i])
                  + "/dMRI/tractography/" + str(list_subjects[i])
                  + "_labels_connectivity_matrix_sift.txt") as file:
            area_sorted = [line.rstrip('\n') for line in file]

            if len(general_list) == 0:
                general_list = area_sorted
            else:
                if not general_list == area_sorted:
                    print(list_subjects[i], ' failed')
                    check_failed = True

    if check_failed:
        print('The labels list is not always the same accross patients')
    else:
        print('Check successful')

    with open(output_path + '_labels_general_list.txt', 'w') as f:
        for line in general_list:
            f.write(str(line) + '\n')


def metrics_analysis(list_subjects: list, root: str, output_path: str, metric_name: list, edge_name: str):

    workbook = xlsxwriter.Workbook(output_path + 'metrics_analysis.xlsx')

    alphabet = list(string.ascii_uppercase)[1:len(metric_name)]

    for i in range(len(edge_name)):

        worksheet = workbook.add_worksheet(str(edge_name))
        worksheet.write('A1', 'Subjects \ Metrics')

        for j in range(len(list_subjects)):

            worksheet.write('A' + str(2 + j), str(list_subjects[j]))

            ROI = tract_to_ROI(
                root + '/subjects/' + list_subjects[j] + '/dMRI/tractography/' + list_subjects[j] + '_' + edge_name[i] + '.trk')

            for k in range(len(metric_name)):

                worksheet.write(str(alphabet[k]) + str(1), metric_name[k])

                if metric_name[k] == 'FA' or metric_name[k] == 'MD' or metric_name[k] == 'RD' or metric_name[k] == 'AD':
                    model = 'dti'
                elif metric_name[k] == 'fintra' or metric_name[k] == 'fextra' or metric_name[k] == 'fiso' or metric_name[k] == 'odi':
                    model = 'noddi'
                elif metric_name[k] == 'wFA' or metric_name[k] == 'wMD' or metric_name[k] == 'wRD' or metric_name[k] == 'wAD' or metric_name[k] == 'diamond_fractions_ftot' or metric_name[k] == 'diamond_fractions_csf':
                    model = 'diamond'
                else:
                    model = 'mf'

                metric_map = nib.load(root + '/subjects/' + list_subjects[j] + '/dMRI/microstructure/'
                                      + model + '/' + list_subjects[j] + '_' + metric_name[k] + '.nii.gz').get_fdata()

                metric_in_ROI = metric_map[ROI != 0]

                mean_ROI = np.mean(metric_in_ROI[metric_in_ROI != 0])

                worksheet.write(str(alphabet[k]) + str(2 + j), mean_ROI)


def mean_metrics_analysis(list_subjects: list, root: str, output_path: str, metric_name: list, edge_name: str):

    workbook = xlsxwriter.Workbook(output_path + 'mean_metrics_analysis.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'Area \ Metrics')

    alphabet = list(string.ascii_uppercase)[1:len(metric_name)]

    for i in range(len(edge_name)):

        worksheet.write('A' + str(2 + i), str(edge_name[i]))

        for j in range(len(metric_name)):

            worksheet.write(str(alphabet[j]) + str(1), metric_name[j])

            mean_list = []

            for k in range(len(list_subjects)):

                ROI = tract_to_ROI(root + '/subjects/' + list_subjects[k]
                                   + '/dMRI/tractography/' + list_subjects[k]
                                   + '_' + edge_name[i] + '.trk')

                if metric_name[j] in ['FA', 'MD', 'RD', 'AD']:
                    model = 'dti'
                elif metric_name[j] in ['fintra', 'fextra', 'fiso', 'odi']:
                    model = 'noddi'
                elif metric_name[j] in ['wFA', 'wMD', 'wRD', 'wAD',
                                        'diamond_fractions_ftot',
                                        'diamond_fractions_csf']:
                    model = 'diamond'
                else:
                    model = 'mf'

                metric_map = nib.load(root + '/subjects/' + list_subjects[k]
                                      + '/dMRI/microstructure/' + model + '/'
                                      + list_subjects[k] + '_' + metric_name[j]
                                      + '.nii.gz').get_fdata()

                metric_in_ROI = metric_map[ROI != 0]

                mean_ROI = np.mean(metric_in_ROI[metric_in_ROI != 0])

                mean_list.append(mean_ROI)

            mean_sub = np.mean(mean_list)

            worksheet.write(str(alphabet[j]) + str(2 + i), mean_ROI)


def labels_matching(excel_path, connectivity_matrix_index_file):

    with open(connectivity_matrix_index_file, 'r') as f:
        area_sorted = [line.rstrip('\n') for line in f]

    df = pd.read_excel(excel_path)

    for i in range(len(df['Area'])):
        for j in range(len(area_sorted)):
            if df.loc[i, 'Area'] == area_sorted[j]:
                df.loc[i, 'Index_new'] = int(j)

    df.to_excel(excel_path.replace('.xlsx', '_bis.xlsx'))


def data_to_json(excel_path: str):

    df = pd.read_excel(excel_path)

    json_path = excel_path.replace('.xlsx', '.json')

    df2 = df.to_json(json_path, orient='columns')
