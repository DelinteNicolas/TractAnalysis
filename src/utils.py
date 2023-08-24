import os
import json
import string
import xlsxwriter
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from unravel.utils import tract_to_ROI
from statannotations.Annotator import Annotator


def to_float64(val):
    """
    Used if *val* is an instance of numpy.float32.
    """

    return np.float64(val)


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

    with open(output_path + 'labels_general_list.txt', 'w') as f:
        for line in general_list:
            f.write(str(line) + '\n')


def metrics_analysis(list_subjects: list, root: str, output_path: str, metric_name: list, edge_name: str):

    with open(edge_name, 'r') as read_file:
        edge_name = json.load(read_file)

    print(edge_name)

    dataframe = {}

    dataframe['Mean'] = {}
    dataframe['Dev'] = {}

    for j, sub in enumerate(list_subjects):

        dataframe['Mean'][sub] = {}
        dataframe['Dev'][sub] = {}

        for i, r in enumerate(edge_name):

            dataframe['Mean'][sub][str(r[0]) + '_' + str(r[1])] = {}
            dataframe['Dev'][sub][str(r[0]) + '_' + str(r[1])] = {}

            ROI = tract_to_ROI(
                root + '/subjects/' + sub + '/dMRI/tractography/tois/' + sub + '_tractogram_sift_' + str(r[0]) + '_' + str(r[1]) + '.trk')

            for k, m in enumerate(metric_name):

                if m in ['FA', 'MD', 'RD', 'AD']:
                    model = 'dti'
                elif m in ['noddi_fintra', 'noddi_fextra', 'noddi_fiso', 'noddi_odi']:
                    model = 'noddi'
                elif m in ['diamond_wFA', 'diamond_wMD', 'diamond_wRD', 'diamond_wAD', 'diamond_fractions_ftot', 'diamond_fractions_csf']:
                    model = 'diamond'
                else:
                    model = 'mf'

                metric_map = nib.load(root + '/subjects/' + sub + '/dMRI/microstructure/'
                                      + model + '/' + sub + '_' + m + '.nii.gz').get_fdata()

                metric_in_ROI = metric_map[ROI != 0]

                mean_ROI = np.mean(metric_in_ROI[metric_in_ROI != 0])
                std_ROI = np.std(metric_in_ROI[metric_in_ROI != 0])

                if(np.isnan(mean_ROI)):
                    mean_ROI = 0
                if(np.isnan(std_ROI)):
                    std_ROI = 0

                dataframe['Mean'][sub][str(r[0]) + '_' + str(r[1])][m] = mean_ROI
                dataframe['Dev'][sub][str(r[0]) + '_' + str(r[1])][m] = std_ROI

    json.dump(dataframe, open(output_path + 'essai_metric_analysis.json', 'w'),
              default=to_float64)

    return output_path + 'essai_metric_analysis.json'

def labels_matching(excel_path, connectivity_matrix_index_file):

    with open(connectivity_matrix_index_file, 'r') as f:
        area_sorted = [line.rstrip('\n') for line in f]

    df = pd.read_excel(excel_path)

    for i in range(len(df['Area'])):
        for j in range(len(area_sorted)):
            if df.loc[i, 'Area'] == area_sorted[j]:
                df.loc[i, 'Index_new'] = int(j)

    df.to_excel(excel_path.replace('.xlsx', '_bis.xlsx'))


def jsonToPandas(jsonFilePath: str):

    import pandas as pd

    file = open(jsonFilePath)
    dic = json.load(file)
    # dic = dic['Mean']
    file.close()

    reform = {(level1_key, level2_key, level3_key, level4_key): values
              for level1_key, level2_dict in dic.items()
              for level2_key, level3_dict in level2_dict.items()
              for level3_key, level4_dict in level3_dict.items()
              for level4_key, values in level4_dict.items()}

    p = pd.DataFrame(reform, index=['Value']).T
    p = p.rename_axis(['Dic', 'Patient', 'Region', 'Metric'])

    return p


def dictionary_without_controls(path_json, control_list):

    data = jsonToPandas(path_json)
    data_value = np.array(data)
    new_data_value = data_value

    indexes = list(data.index)
    for i, j in enumerate(indexes):
        indexes[i] = np.array(j)
    indexes = np.array(indexes)

    new_indexes = indexes
    list_subj = new_indexes[:, 1]

    for i in range(len(list_subj) - 1, -1, -1):
        if list_subj[i] in control_list:
            new_data_value = np.delete(new_data_value, i, axis=0)
            new_indexes = np.delete(new_indexes, i, axis=0)

    dataframe = {}
    for i in range(len(new_data_value)):
        dataframe[str(new_indexes[i, 0]), str(new_indexes[i, 1]), str(new_indexes[i, 2]), str(new_indexes[i, 3])] = new_data_value[i][0]
    dataframe2 = pd.DataFrame(dataframe, index=['Value']).T
    dataframe2 = dataframe2.rename_axis(['Dic', 'Patient', 'Region', 'Metric'])
    dataframe2 = dataframe2.sort_values(by='Dic', ascending=False)

    return dataframe2


def dictionary_controls(path_json, control_list):

    data = jsonToPandas(path_json)
    data_value = np.array(data)
    new_data_value = data_value

    indexes = list(data.index)
    for i, j in enumerate(indexes):
        indexes[i] = np.array(j)
    indexes = np.array(indexes)

    new_indexes = indexes
    list_subj = new_indexes[:, 1]

    for i in range(len(list_subj) - 1, -1, -1):
        if list_subj[i] not in control_list:
            new_data_value = np.delete(new_data_value, i, axis=0)
            new_indexes = np.delete(new_indexes, i, axis=0)

    dataframe = {}

    for i in range(len(new_data_value)):
        dataframe[str(new_indexes[i, 0]), str(new_indexes[i, 1]), str(new_indexes[i, 2]), str(new_indexes[i, 3])] = new_data_value[i][0]

    dataframe2 = pd.DataFrame(dataframe, index=['Value']).T
    dataframe2 = dataframe2.rename_axis(['Dic', 'Patient', 'Region', 'Metric'])
    dataframe2 = dataframe2.sort_values(by='Dic', ascending=False)

    return dataframe2


def list_nom_patient(dataframe, control_list):

    patients = dataframe.index.unique(1)
    name_controls = []
    name_patients = []

    for i in patients:
        if i in control_list:
            i = i.replace('_E1', '')
            i = i.replace('_E2', '')
            i = i.replace('_E3', '')

            if i not in name_controls:
                name_controls.append(i)
        else:
            i = i.replace('_E1', '')
            i = i.replace('_E2', '')
            i = i.replace('_E3', '')

            if i not in name_patients:
                name_patients.append(i)

    return name_patients, name_controls


def difference_temps(dataframe, control_list, dic_val):

    name_patients, name_controls = list_nom_patient(dataframe, control_list)
    name_all = np.append(name_patients, name_controls)

    df_temps = {}
    list_temps = []

    for t in list(dataframe.index.unique(1)):
        if t[-2:] not in list_temps:
            list_temps.append(t[-2:])

    list_temps.sort()

    for region_val in list(dataframe.index.unique(2)):
        for metric_val in list(dataframe.index.unique(3)):
            for i in name_all:
                inter_value = []
                for temps in list_temps:
                    try:
                        inter_value.append(dataframe.loc[dic_val, i + '_' + temps, region_val, metric_val][0])
                    except KeyError:
                        inter_value.append(float('nan'))
                        continue

                df_temps[dic_val, i, region_val, metric_val] = inter_value

    temps_dataframe = pd.DataFrame(df_temps, index=[list_temps]).T
    temps_dataframe = temps_dataframe.rename_axis(['Dic', 'Patient', 'Region', 'Metric'])
    temps_dataframe = temps_dataframe.sort_values(by='Patient', ascending=True)

    temps_dataframe['Diff E2-E1'] = np.array(temps_dataframe.loc[:, 'E2']) - np.array(temps_dataframe.loc[:, 'E1'])
    temps_dataframe['Diff E3-E1'] = np.array(temps_dataframe.loc[:, 'E3']) - np.array(temps_dataframe.loc[:, 'E1'])
    temps_dataframe['Diff E3-E2'] = np.array(temps_dataframe.loc[:, 'E3']) - np.array(temps_dataframe.loc[:, 'E2'])

    return temps_dataframe


def graphs_analysis(path_json, specific, region, control_list, dic=None, metric=None):

    data = jsonToPandas(path_json)
    data_value = np.array(data)
    new_data_value = data_value

    indexes = list(data.index)
    for i, j in enumerate(indexes):
        indexes[i] = np.array(j)
    indexes = np.array(indexes)

    new_indexes = indexes
    list_subj = new_indexes[:, 1]

    for i in range(len(list_subj) - 1, -1, -1):
        if list_subj[i] in control_list:
            new_data_value = np.delete(new_data_value, i, axis=0)
            new_indexes = np.delete(new_indexes, i, axis=0)

    dataframe = {}
    for i in range(len(new_data_value)):
        dataframe[str(new_indexes[i, 0]), str(new_indexes[i, 1]), str(new_indexes[i, 2]), str(new_indexes[i, 3])] = new_data_value[i][0]
    dataframe2 = pd.DataFrame(dataframe, index=['Value']).T
    dataframe2 = dataframe2.rename_axis(['Dic', 'Patient', 'Region', 'Metric'])
    dataframe2 = dataframe2.sort_values(by='Dic', ascending=False)

    temps_list = list(dataframe2.index.unique(1))
    for i, j in enumerate(temps_list):
        if "E1" in j and j not in control_list:
            temps_list[i] = "E1"
        elif "E2" in j and j not in control_list:
            temps_list[i] = "E2"
        elif "E3" in j and j not in control_list:
            temps_list[i] = "E3"

    if specific:
        assert dic != None, 'To compute the graphs, the dic variable must have a value set to something other than None'
        assert metric != None, 'To compute the graphs, the metric variable must have a value set to something other than None'

        df = pd.DataFrame({'Temps': temps_list, 'Value': np.squeeze(dataframe2.loc[dic, :, region, metric])})

        fig = plt.figure()
        ax = sns.violinplot(x='Temps', y='Value', data=df, order=['E1', 'E2', 'E3'])
        sns.stripplot(x="Temps", y="Value", data=df, jitter=True, order=['E1', 'E2', 'E3'], color='w')
        plt.ylabel(metric)
        plt.title(metric + ' values of all patient for the edge ' + region)

        pairs = [('E1', 'E2'), ('E1', 'E3'), ('E3', 'E2')]
        annotator = Annotator(ax, pairs, data=df, x='Temps', y='Value', order=['E1', 'E2', 'E3'])
        annotator.configure(test="t-test_ind", text_format='simple', loc='inside')
        a = annotator.apply_and_annotate()
        plt.tight_layout()
        plt.show()

    else:
        pval = {}

        for dic_val in list(dataframe2.index.unique(0)):
            for region_val in list(dataframe2.index.unique(2)):
                for metric_val in list(dataframe2.index.unique(3)):
                    df = pd.DataFrame({'Temps': temps_list, 'Value': np.squeeze(dataframe2.loc[dic_val, :, region_val, metric_val])})

                    fig = plt.figure()
                    ax = sns.violinplot(x='Temps', y='Value', data=df, order=['E1', 'E2', 'E3'])
                    plt.ylabel(metric)
                    plt.title(dic_val + ' ' + metric_val + ' values of all patient for the edge ' + region_val)

                    pairs = [('E1', 'E2'), ('E2', 'E3'), ('E1', 'E3')]
                    annotator = Annotator(ax, pairs, data=df, x='Temps', y='Value', order=['E1', 'E2', 'E3'])
                    annotator.configure(test="t-test_welch", text_format='simple', loc='inside')
                    a = annotator.apply_and_annotate()

                    pval[dic_val, region_val, metric_val] = np.array([a[1][0].data.pvalue, a[1][1].data.pvalue, a[1][2].data.pvalue])

        pval_dataframe = pd.DataFrame(pval, index=['E1-E2', 'E2-E3', 'E1-E3']).T
        pval_dataframe = pval_dataframe.rename_axis(['Dic', 'Region', 'Metric'])
        pval_dataframe = pval_dataframe.sort_values(by='Dic', ascending=False)
