import os
import json
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from collections import defaultdict, OrderedDict
from itertools import combinations, groupby
from dipy.tracking.utils import ndbincount


def to_float64(val):
    '''
    Used if *val* is an instance of numpy.float32.
    '''

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
        return 'axial'
    elif (d != 0 and e != 0 and f != 0 and sum_diag_aff == 0):
        return 'sagittal'
    elif (g != 0 and h != 0 and i != 0 and sum_diag_aff == 0):
        return 'coronal'
    else:
        return 'oblique'


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

# %% Cell 2 - Modified from DIPY
def connectivity_matrix(streamlines, affine, label_volume, inclusive=False,
                        symmetric=True, return_mapping=False,
                        mapping_as_streamlines=False):
    '''
    Count the streamlines that start and end at each label pair.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    label_volume : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    inclusive: bool
        Whether to analyze the entire streamline, as opposed to just the
        endpoints. Allowing this will increase calculation time and mapping
        size, especially if mapping_as_streamlines is True. False by default.
    symmetric : bool, True by default
        Symmetric means we don't distinguish between start and end points. If
        symmetric is True, ``matrix[i, j] == matrix[j, i]``.
    return_mapping : bool, False by default
        If True, a mapping is returned which maps matrix indices to
        streamlines.
    mapping_as_streamlines : bool, False by default
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.

    Returns
    -------
    matrix : ndarray
        The number of connection between each pair of regions in
        `label_volume`.
    mapping : defaultdict(list)
        ``mapping[i, j]`` returns all the streamlines that connect region `i`
        to region `j`. If `symmetric` is True mapping will only have one key
        for each start end pair such that if ``i < j`` mapping will have key
        ``(i, j)`` but not key ``(j, i)``.

    '''
    # Error checking on label_volume

    print('Using this shit')
    kind = label_volume.dtype.kind
    labels_positive = ((kind == 'u')
                       or ((kind == 'i') and (label_volume.min() >= 0)))
    valid_label_volume = (labels_positive and label_volume.ndim == 3)
    if not valid_label_volume:
        raise ValueError('label_volume must be a 3d integer array with'
                         'non-negative label values')

    # If streamlines is an iterator
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)

    if inclusive:
        # Create ndarray to store streamline connections
        edges = np.ndarray(shape=(3, 0), dtype=int)
        # lin_T, offset = _mapping_to_voxel(affine)
        for sl, _ in enumerate(streamlines):
            # Convert streamline to voxel coordinates
            # entire = _to_voxel_coordinates(streamlines[sl], lin_T, offset)
            i, j, k = streamlines[sl].T

            if symmetric:
                # Create list of all labels streamline passes through
                entirelabels = list(OrderedDict.fromkeys(label_volume[i, j, k]))
                # Append all connection combinations with streamline number
                for comb in combinations(entirelabels, 2):
                    edges = np.append(edges, [[comb[0]], [comb[1]], [sl]],
                                      axis=1)
            else:
                # Create list of all labels streamline passes through, keeping
                # order and whether a label was entered multiple times
                entirelabels = list(groupby(label_volume[i, j, k]))
                # Append connection combinations along with streamline number,
                # removing duplicates and connections from a label to itself
                combs = set(combinations([z[0] for z in entirelabels], 2))
                for comb in combs:
                    if comb[0] == comb[1]:
                        pass
                    else:
                        edges = np.append(edges, [[comb[0]], [comb[1]], [sl]],
                                          axis=1)
        if symmetric:
            edges[0:2].sort(0)
        mx = label_volume.max() + 1
        matrix = ndbincount(edges[0:2], shape=(mx, mx))

        if symmetric:
            matrix = np.maximum(matrix, matrix.T)
        if return_mapping:
            mapping = defaultdict(list)
            for i, (a, b, c) in enumerate(edges.T):
                mapping[a, b].append(c)
            # Replace each list of indices with the streamlines they index
            if mapping_as_streamlines:
                for key in mapping:
                    mapping[key] = [streamlines[i] for i in mapping[key]]

            return matrix, mapping

        return matrix
    else:
        # take the first and last point of each streamline
        endpoints = [sl[0::len(sl) - 1] for sl in streamlines]

        # Map the streamlines coordinates to voxel coordinates
        # lin_T, offset = _mapping_to_voxel(affine)
        # endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

        # get labels for label_volume

        i, j, k = np.array(endpoints).astype(dtype=int).T

        endlabels = label_volume[i, j, k]

        if symmetric:
            endlabels.sort(0)
        mx = label_volume.max() + 1
        matrix = ndbincount(endlabels, shape=(mx, mx))
        if symmetric:
            matrix = np.maximum(matrix, matrix.T)

        if return_mapping:
            mapping = defaultdict(list)
            for i, (a, b) in enumerate(endlabels.T):
                mapping[a, b].append(i)

            # Replace each list of indices with the streamlines they index
            if mapping_as_streamlines:
                for key in mapping:
                    mapping[key] = [streamlines[i] for i in mapping[key]]

            # Return the mapping matrix and the mapping
            return matrix, mapping

        return matrix


# %% Cell 3 - Verification of labels
def check_labels(list_subjects: str, root: str, output_path: str):

    general_list = []
    check_failed = False

    # with open(list_subjects, 'r') as read_file:
    #     list_subjects = json.load(read_file)

    for i in range(len(list_subjects)):

        with open(root + 'subjects/' + str(list_subjects[i])
                  + '/dMRI/tractography/' + str(list_subjects[i])
                  + '_labels_connectivity_matrix_sift.txt') as file:
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
        for j in range(1, len(area_sorted) + 1):
            if df.loc[i, 'Area'] == area_sorted[j - 1]:
                df.loc[i, 'Index_new'] = int(j)

    df.to_excel(excel_path.replace('.xlsx', '_bis.xlsx'))


# %% Cell 4 - Computing mean connectivity
def get_min_connectivity(output_path: str, evolution: bool, list_temps: list = []):
    '''
    Return the minimum connectivity matrix.

    Parameters
    ----------
    output_path : str
        DESCRIPTION.
    evolution : bool
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

    if evolution and len(list_temps) > 0:
        list_E1 = np.load(output_path + 'list_E1.npy')
        list_E2 = np.load(output_path + 'list_E2.npy')

        list_matrices = np.append(list_E1, list_E2, axis=2)

        min_connectivity = np.min(list_matrices, axis=2)

        np.save(output_path + 'min_connectivity_matrix_evolution.npy', min_connectivity)

    else:
        list_E1 = np.load(output_path + 'list_E1.npy')
        list_E2 = np.load(output_path + 'list_E2.npy')
        list_E3 = np.load(output_path + 'list_E3.npy')

        list_matrices = np.append(np.append(list_E1, list_E2, axis=2), list_E3, axis=2)

        min_connectivity = np.min(list_matrices, axis=2)

        np.save(output_path + 'min_connectivity_matrix.npy', min_connectivity)


# %% Cell 5 - Dictionary
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


# %% Cell 6 - Graphs
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
