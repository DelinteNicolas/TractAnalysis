import os
import json
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation


def to_float64(val):
    """
    Used if *val* is an instance of numpy.float32.
    """

    return np.float64(val)

def add_regions(edge_name, new_regions):

    for i in range(len(new_regions)):
        edge_name.append([new_regions[i]])

    return edge_name

# %% Cell 1 - Checking view orientations for all patients
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


# %% Cell 2 - Verification of labels
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

def labels_matching(excel_path, connectivity_matrix_index_file):

    with open(connectivity_matrix_index_file, 'r') as f:
        area_sorted = [line.rstrip('\n') for line in f]

    df = pd.read_excel(excel_path)

    for i in range(len(df['Area'])):
        for j in range(len(area_sorted)):
            if df.loc[i, 'Area'] == area_sorted[j]:
                df.loc[i, 'Index_new'] = int(j)

    df.to_excel(excel_path.replace('.xlsx', '_bis.xlsx'))


# %% Cell 3 - Computing mean connectivity
def get_min_connectivity(output_path: str, evolution: bool):
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

    if os.path.isfile(output_path + 'labels_connectivity_matrix.txt'):
        labels_found = True
        with open(output_path + 'labels_connectivity_matrix.txt', 'r') as f:
            area_sorted = [line.rstrip('\n') for line in f]
    else:
        labels_found = False

    if evolution:
        list_matrices = np.load(output_path + 'evolution_patient.npy')

        min_connectivity = np.min(abs(list_matrices), axis=2)

        np.save(output_path + 'min_connectivity_matrix_evolution.npy', min_connectivity)

    else:
        list_E1 = np.load(output_path + 'list_E1.npy')
        list_E2 = np.load(output_path + 'list_E2.npy')
        list_E3 = np.load(output_path + 'list_E3.npy')

        list_matrices = np.append(np.append(list_E1, list_E2, axis=2), list_E3, axis=2)

        min_connectivity = np.min(list_matrices, axis=2)

        np.save(output_path + 'min_connectivity_matrix.npy', min_connectivity)


# %% Cell 4 - Dictionary
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


def difference_temps(dic_val, patient_list, output_path):

    dataframe = jsonToPandas(output_path + 'metric_analysis.json')

    name_all = [x.replace('_E1', '').replace('_E2', '').replace('_E3', '') for x in patient_list]

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
                        inter_value.append(
                            dataframe.loc[dic_val, i + '_' + temps, region_val,
                                          metric_val][0])
                    except KeyError:
                        inter_value.append(float('nan'))
                        continue

                df_temps[dic_val, i, region_val, metric_val] = inter_value

    temps_dataframe = pd.DataFrame(df_temps, index=[list_temps]).T
    temps_dataframe = temps_dataframe.rename_axis(
        ['Dic', 'Patient', 'Region', 'Metric'])
    temps_dataframe = temps_dataframe.sort_values(by='Patient', ascending=True)

    temps_dataframe['Diff E2-E1'] = (np.array(temps_dataframe.loc[:, 'E2']) - np.array(temps_dataframe.loc[:, 'E1']))
    temps_dataframe['Diff E3-E1'] = (np.array(temps_dataframe.loc[:, 'E3']) - np.array(temps_dataframe.loc[:, 'E1']))
    temps_dataframe['Diff E3-E2'] = (np.array(temps_dataframe.loc[:, 'E3']) - np.array(temps_dataframe.loc[:, 'E2']))

    temps_dataframe['Change E2-E1 (%)'] = (np.array(temps_dataframe.loc[:, 'E2']) - np.array(temps_dataframe.loc[:, 'E1'])) * 100 / np.array(temps_dataframe.loc[:, 'E1'])
    temps_dataframe['Change E3-E1 (%)'] = (np.array(temps_dataframe.loc[:, 'E3']) - np.array(temps_dataframe.loc[:, 'E1'])) * 100 / np.array(temps_dataframe.loc[:, 'E1'])
    temps_dataframe['Change E3-E2 (%)'] = (np.array(temps_dataframe.loc[:, 'E3']) - np.array(temps_dataframe.loc[:, 'E2'])) * 100 / np.array(temps_dataframe.loc[:, 'E2'])

    # temps_dataframe.to_json(output_path + 'difference_time.json')


def dictionary_patients(path_json, control_list):

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
        dataframe[str(new_indexes[i, 0]), str(new_indexes[i, 1]), str(
            new_indexes[i, 2]), str(new_indexes[i, 3])] = new_data_value[i][0]
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
        dataframe[str(new_indexes[i, 0]), str(new_indexes[i, 1]), str(
            new_indexes[i, 2]), str(new_indexes[i, 3])] = new_data_value[i][0]

    dataframe2 = pd.DataFrame(dataframe, index=['Value']).T
    dataframe2 = dataframe2.rename_axis(['Dic', 'Patient', 'Region', 'Metric'])
    dataframe2 = dataframe2.sort_values(by='Dic', ascending=False)

    return dataframe2


# %% Cell 5 - Graphs
def graphs_analysis(dataframe, region, dic, metric, temps_list):

    dataframe = dataframe.loc[dic, :, region, metric]
    dataframe = dataframe.sort_values(by='Patient', ascending=True)

    fig, ax = plt.subplots()
    ax = sns.violinplot(data=dataframe, order=temps_list,
                        palette=sns.color_palette('pastel'))
    sns.stripplot(data=dataframe, jitter=True, order=temps_list,
                  palette=sns.color_palette(), ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(metric + ' values of all patient for the edge ' + region)

    if len(temps_list) == 3:
        pairs = [('E1', 'E2'), ('E1', 'E3'), ('E3', 'E2')]
        nums = [0, 1, 2]
    else:
        pairs = [('E1', 'E2')]
        nums = [0, 1]

    a = add_stat_annotation(ax, data=dataframe, order=temps_list,
                            box_pairs=pairs,
                            test='t-test_ind', text_format='star', verbose=2)

    maximal_val = np.max(np.max(dataframe)) + 0.5 * \
        (np.max(np.max(dataframe)) - np.min(np.min(dataframe)))
    minimal_val = np.min(np.min(dataframe)) - 0.5 * \
        (np.max(np.max(dataframe)) - np.min(np.min(dataframe)))

    ax.set_xticks(nums)
    ax.set_xlim(nums[0] - 0.5, nums[-1] + 0.5)
    ax.set_ylim(minimal_val, maximal_val)
    plt.tight_layout()
    plt.show()

    return a
