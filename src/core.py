import os
import json
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from utils import add_regions, to_float64, connectivity_matrix
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from unravel.utils import tensor_to_DTI, tract_to_ROI
from unravel.core import (get_fixel_weight, get_microstructure_map,
                          get_weighted_mean, tensor_to_peak)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from regis.core import find_transform, apply_transform
    from dipy.io.streamline import load_tractogram, save_trk
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    import dipy.tracking

# %% Cell 1 - SLurm Iter
def slurm_iter(root: str, code: str, patient_list: list = []):
    '''
    Launches the scripts.py python file for all patients in patient_list

    Parameters
    ----------
    root : str
        Path to study
    code : str
        Part of the script to launch. Either 'connectivity' or 'extraction'.
    patient_list : list, optional
        If not specified, runs on all patients in the study. The default is [].

    Returns
    -------
    None.

    '''

    if len(patient_list) == 0:
        patient_list = json.load(open(root + 'subjects/subj_list.json', 'r'))

    path_to_analysis_code = root.replace(
        root.split('/')[-2] + '/', '') + 'TractAnalysis/'
    path_to_code = path_to_analysis_code + 'src/scripts.py'

    for patient in patient_list:

        os.system('sbatch -J ' + patient + ' '
                  + path_to_analysis_code + 'slurm/submitIter.sh '
                  + path_to_code + ' ' + patient + ' ' + root + ' ' + code)


# %% Cell 2 - Sending labels to mni space
def register_labels_to_atlas(labels_path: str, mni_fa_path: str, output_path: str):
    '''
    Sending labels to mni space.

    Parameters
    ----------
    labels_path : str
        Moving file.
    mni_fa_path : str
        Static file.
    output_path : str
        Path where the labels are stored.

    Returns
    -------
    None.

    '''

    map_desikan_to_fa = find_transform(labels_path, mni_fa_path, only_affine=True)

    apply_transform(labels_path, map_desikan_to_fa, static_file=mni_fa_path, output_path=output_path, labels=True)


# %% Cell 3 - Registration of the atlas in the patient space
def register_atlas_to_subj(fa_path: str, label_path: str, mni_fa_path: str,
                           output_path: str, static_mask_path: str):
    '''
    Two-step registration to obtain label in the diffusion space of each patient.

    Parameters
    ----------
    fa_path : str
        Path to FA file (static file).
    label_path : str
        Label registered, moving file.
    mni_fa_path : str
        DESCRIPTION.
    output_path : str
        DESCRIPTION.
    static_mask_path : str
        Static mask that defines which pixels in the static image are not set to 0.

    Returns
    -------
    None.

    '''

    static_mask = nib.load(static_mask_path).get_fdata()
    map_mni_to_subj = find_transform(mni_fa_path, fa_path, hard_static_mask=static_mask)

    apply_transform(label_path, map_mni_to_subj, static_file=fa_path, output_path=output_path, labels=True)


# %% Cell 4 - Connectivity matrices
def check_wanted(unwanted_keyword_list: list, long_name: str) -> bool:
    '''


    Parameters
    ----------
    unwanted_keyword_list : list
        DESCRIPTION.
    long_name : str
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    '''

    for key in unwanted_keyword_list:
        if key in long_name:
            return False

    return True


def connectivity_matrices(dwi_path: str, labels_path: str,
                          streamlines_path: str, output_path: str,
                          freeSurfer_labels: str):
    '''
    Creation of the connectivity matrix for each patient at each acquisition
    time.

    Parameters
    ----------
    dwi_path : str
        DESCRIPTION.
    labels_path : str
        DESCRIPTION.
    streamlines_path : str
        DESCRIPTION.
    output_path : str
        DESCRIPTION.
    freeSurfer_labels : str
        DESCRIPTION.
    subjects_list : str
        DESCRIPTION.

    Returns
    -------
    new_label_map : TYPE
        DESCRIPTION.

    '''

    labels = nib.load(labels_path).get_fdata()

    trk = load_tractogram(streamlines_path, 'same')
    trk.to_vox()

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

    unwanted = ['vessel', 'CSF', 'Vent', 'Unknown', 'White_Matter', 'WM',
                'Chiasm']

    for i in range(len(values)):
        for j in range(len(df['Index'])):
            wanted = check_wanted(unwanted, str(df['Area'][j]))
            if (values[i] == df['Index'][j]) and wanted:
                if 'right' in str(df['Area'][j]):
                    right_labels.append(values[i])
                    right_area.append(df['Area'][j])
                elif 'left' in str(df['Area'][j]):
                    left_labels.append(values[i])
                    left_area.append(df['Area'][j])
                else:
                    middle_labels.append(values[i])
                    middle_area.append(df['Area'][j])

    labels_sorted = np.append(
        np.append(right_labels, middle_labels), left_labels)
    a = np.argwhere(labels_sorted == 0)

    labels_sorted = np.delete(labels_sorted, a)
    area_sorted = np.append(np.append(right_area, middle_area), left_area)
    area_sorted = np.delete(area_sorted, a)

    new_labels_value = np.linspace(
        0, len(labels_sorted) - 1, len(labels_sorted)) + 1
    new_label_map = np.zeros([len(labels), len(labels[0]), len(labels[0][0])])
    for i in range(len(labels_sorted)):
        new_label_map += np.where(labels
                                  == labels_sorted[i], new_labels_value[i], 0)
    new_label_map = new_label_map.astype(np.uint8)

    M, _ = connectivity_matrix(streams_data, affine,
                               new_label_map,
                               return_mapping=True,
                               mapping_as_streamlines=True)

    np.fill_diagonal(M, 0)
    M = M[1:, 1:]
    M = M.astype('float32')
    M = M / np.sum(M)

    fig, ax = plt.subplots()
    ax.imshow(np.log1p(M * len(trk.streamlines._offsets)))
    ax.set_yticks(np.arange(len(area_sorted)))
    ax.set_yticklabels(area_sorted)

    trac = streamlines_path.replace('_tractogram', '_connectivity_matrix')
    trac_im = trac.replace('.trk', '.png')

    plt.savefig(trac_im)
    plt.title('Connectivity matrix')
    plt.xlabel('Labels')

    labels_path = streamlines_path.replace(
        '_tractogram', '_labels_connectivity_matrix')
    labels_path = labels_path.replace('.trk', '.txt')

    with open(labels_path, 'w') as f:
        for line in area_sorted:
            f.write(str(line) + '\n')

    trac_save = trac.replace('.trk', '.npy')
    np.save(trac_save, M)

    return new_label_map


# %% Cell 5 - Computing p-values of connectivity matrices
def significance_level(list_subject, root: str, output_path: str):
    '''


    Parameters
    ----------
    list_subject : str
        DESCRIPTION.
    root : str
        DESCRIPTION.
    output_path : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    if list_subject is str:
        with open(list_subject, 'r') as read_file:
            subj_list = json.load(read_file)
    else:
        subj_list = list_subject

    list_E1 = []
    list_E2 = []
    list_E3 = []

    for nombre_i, sub in enumerate(subj_list):

        path = (root + 'subjects/' + str(sub) + '/dMRI/tractography/' + str(sub) + '_connectivity_matrix_sift.npy')
        try:
            matrix = np.load(path)
        except FileNotFoundError:
            print('Connectivity matrix not found for ' + str(sub))
            continue

        if 'E1' in str(sub):
            list_E1.append(matrix)
        elif 'E2' in str(sub):
            list_E2.append(matrix)
        elif 'E3' in str(sub):
            list_E3.append(matrix)

    list_E1 = np.stack(list_E1, axis=2)
    list_E2 = np.stack(list_E2, axis=2)
    list_E3 = np.stack(list_E3, axis=2)

    np.save(output_path + 'list_E1.npy', list_E1)
    np.save(output_path + 'list_E2.npy', list_E2)
    np.save(output_path + 'list_E3.npy', list_E3)

    _, pval_12 = ttest_ind(list_E1, list_E2, axis=2, alternative='two-sided', nan_policy='omit', equal_var=False)
    pval_12[np.isnan(pval_12)] = 1

    _, pval_13 = ttest_ind(list_E1, list_E3, axis=2, alternative='two-sided', nan_policy='omit', equal_var=False)
    pval_13[np.isnan(pval_13)] = 1

    _, pval_23 = ttest_ind(list_E2, list_E3, axis=2, alternative='two-sided', nan_policy='omit', equal_var=False)
    pval_23[np.isnan(pval_23)] = 1

    pval_all = np.stack([pval_12, pval_13, pval_23], axis=2)

    np.save(output_path + 'pvals_E12_E13_E23.npy', pval_all)


def significance_level_evolution(subj_list, control_list, root, output_path):

    evolution_patient = []
    evolution_control = []

    list_subj = [x.replace('_E1', '').replace('_E2', '').replace('_E3', '') for x in subj_list]
    list_control = [x.replace('_E1', '').replace('_E2', '') for x in control_list]

    list_subj = np.unique(list_subj)
    list_control = np.unique(list_control)

    for sub in list_subj:

        path_E1 = (root + 'subjects/' + str(sub) + '_E1' + '/dMRI/tractography/' + str(sub) + '_E1' + '_connectivity_matrix_sift.npy')
        path_E2 = (root + 'subjects/' + str(sub) + '_E2' + '/dMRI/tractography/' + str(sub) + '_E2' + '_connectivity_matrix_sift.npy')

        bool_E1 = False
        bool_E2 = False

        if str(sub) + '_E1' in subj_list:
            matrix_E1 = np.load(path_E1)
        else:
            print('Connectivity matrix not found for ' + str(sub) + '_E1')
            bool_E1 = True

        if str(sub) + '_E2' in subj_list:
            matrix_E2 = np.load(path_E2)
        else:
            print('Connectivity matrix not found for ' + str(sub) + '_E2')
            bool_E2 = True

        if bool_E1:
            evolution_patient.append(np.zeros((matrix_E2.shape)) * np.nan)
        elif bool_E2:
            evolution_patient.append(np.zeros((matrix_E1.shape)) * np.nan)
        else:
            evolution_patient.append(matrix_E2 - matrix_E1)

    for cont in list_control:

        control_E1 = (root + 'subjects/' + str(cont) + '_E1' + '/dMRI/tractography/' + str(cont) + '_E1' + '_connectivity_matrix_sift.npy')
        control_E2 = (root + 'subjects/' + str(cont) + '_E2' + '/dMRI/tractography/' + str(cont) + '_E2' + '_connectivity_matrix_sift.npy')

        bool_E1 = False
        bool_E2 = False

        if str(cont) + '_E1' in control_list:
            matrice_E1 = np.load(control_E1)
        else:
            print('Connectivity matrix not found for ' + str(cont) + '_E1')
            bool_E1 = True

        if str(cont) + '_E2' in control_list:
            matrice_E2 = np.load(control_E2)
        else:
            print('Connectivity matrix not found for ' + str(cont) + '_E2')
            bool_E2 = True

        if bool_E1:
            evolution_control.append(np.zeros((matrice_E2.shape)) * np.nan)
        elif bool_E2:
            evolution_control.append(np.zeros((matrice_E1.shape)) * np.nan)
        else:
            evolution_control.append(matrice_E2 - matrice_E1)

    evolution_patient = np.stack(evolution_patient, axis=2)
    evolution_control = np.stack(evolution_control, axis=2)

    np.save(output_path + 'evolution_patient.npy', evolution_patient)
    np.save(output_path + 'evolution_control.npy', evolution_control)

    _, pval_12 = ttest_ind(evolution_patient, evolution_control, axis=2, alternative='two-sided', nan_policy='omit', equal_var=False)
    pval_12[np.isnan(pval_12)] = 1

    a = np.array(pval_12)

    np.save(output_path + 'pvals_E12.npy', a)

# %% Cell 6 - Finding most relevant connectivity edges
def get_edges_of_interest(pval_file: str, output_path: str,
                          min_path: str) -> list:
    '''
    Returns the edges corresponding to low p-values

    Parameters
    ----------
    pval_file : str
        Path to .npy containing p-values of connectivity matrices.
    output_path :str
        ...
    min_path :str
        Path to arrays with the minimum number of connections.

    Returns
    -------
    list
        List of tuples containing the index of the regions cennected by the edge
        interest

    '''

    pval = np.load(pval_file)

    # Removing duplicates due to symmetry
    pval = np.transpose(pval)
    pval = np.tril(pval, k=0)
    pval[pval == 0] = 1
    pval = np.transpose(pval)

    # Removing regions where there are no connections
    mins = np.load(min_path)
    pval[mins == 0] = 1

    if 'E23' in pval_file:
        comparisons = np.count_nonzero(pval != 1) / pval.shape[2]
    else:
        comparisons = np.count_nonzero(pval != 1)

    print(comparisons)

    pval_tresh = 0.05
    selec = np.argwhere(pval < pval_tresh)
    print('Number of significant values found: ', len(selec))

    # Multiple comparison pval correction
    # Bonferroni --------------------------------------

    pval_tresh = 0.05 / comparisons
    selec = np.argwhere(pval < pval_tresh)
    print(selec)
    print('Number of significant values found with Bonferroni: ', len(selec))

    if len(selec) < 5:

        # Benjamini-Hochberg ------------------------------

        pval_cand = np.sort(pval[pval != 1])
        pval_cand_copy = pval_cand.copy()

        # False discovery rate
        Q = .2

        for i, p in enumerate(pval_cand):
            if p > (i + 1) / comparisons * Q:
                pval_cand[i] = 1

        selec = np.argwhere(np.isin(pval, pval_cand[pval_cand != 1]))

        print('Number of significant values found with Benjamini: ', len(selec))

    if len(selec) < 5:

        # Temporary candidate ------------------------------
        selec = np.argwhere(np.isin(pval, pval_cand_copy[:10]))
        print('Minimum p-values used instead of multiple correction')

    # First values of candidate pvalues
    edges = []
    for i in range(selec.shape[0]):

        edge = (int(selec[i, 0]), int(selec[i, 1]))
        if edge not in edges:
            edges.append(edge)

    if 'E23' in pval_file:
        json.dump(edges, open(output_path + 'selected_edges.json', 'w'))
    else:
        json.dump(edges, open(output_path + 'selected_edges_evolution.json', 'w'))


# %% Cell 7 - Extract tract of interest
def extract_streamline(edge: tuple, labels_path: str,
                       streamlines_path: str, excel_path: str):
    '''
    Creates a new file with the streamlines connecting both regions specified in
    the tuple 'edge'.

    Parameters
    ----------
    edge : tuple
        Index of the regions of interest. Ex: (1,23)
    labels_path : str
        Path to volume containing the indexes.
    streamlines_path : str
        Path to tractogram.

    Returns
    -------
    None.

    '''

    labels = nib.load(labels_path).get_fdata()

    img = nib.load(labels_path)
    affine = img.affine

    trk = load_tractogram(streamlines_path, 'same')
    # trk.to_vox()
    # trk.to_corner()
    streamlines = trk.streamlines

    df = pd.read_excel(excel_path)

    mask1 = np.where(
        labels == df.loc[df['Index_new'] == edge[0]]['Index'].iloc[0], 1, 0)
    mask2 = np.where(
        labels == df.loc[df['Index_new'] == edge[1]]['Index'].iloc[0], 1, 0)

    streamlines = dipy.tracking.utils.target(streamlines, affine, mask1, include=True)
    streamlines = dipy.tracking.utils.target(streamlines, affine, mask2, include=True)

    tract = StatefulTractogram(streamlines, img, Space.RASMM)

    output_path = (streamlines_path.replace(streamlines_path.split('/')[-1], '')
                   + 'tois/')

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    filename = (output_path + streamlines_path.split('/')[-1][:-4] + '_'
                + str(edge[0]) + '_' + str(edge[1]))

    save_trk(tract, filename + '.trk')


# %% Cell 8 - Mean tract microstructure metrics
def create_tensor_metrics(path: str):
    '''


    Parameters
    ----------
    path : str
        Ex: '/.../diamond/subjectName'

    Returns
    -------
    None.

    '''

    for tensor in ['t0', 't1']:

        img = nib.load(path + '_diamond_' + tensor + '.nii.gz')
        t = img.get_fdata()

        FA, AD, RD, MD = tensor_to_DTI(t)

        metric = {}
        metric['FA'] = FA
        metric['MD'] = MD
        metric['AD'] = AD
        metric['RD'] = RD

        for m in metric:
            out = nib.Nifti1Image(metric[m].real, img.affine)
            out.header.get_xyzt_units()
            out.to_filename(path + '_diamond_' + m + '_' + tensor + '.nii.gz')


def get_mean_tracts(trk_file: str, micro_path: str):
    '''
    Return means for all metrics for a single patient using UNRAVEL

    Parameters
    ----------
    trk_file : str
        DESCRIPTION.
    micro_path : str
        Patient specific path to microstructure folder

    Returns
    -------
    mean : TYPE
        DESCRIPTION.
    dev : TYPE
        DESCRIPTION.

    '''

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()

    subject = micro_path.split('/')[-4]

    mean_dic = {}
    dev_dic = {}

    # Streamline count ---------------------

    mean_dic['stream_count'] = len(trk.streamlines._offsets)
    dev_dic['stream_count'] = 0

    # Diamond ------------------------------

    tensor_files = [micro_path + 'diamond/' + subject + '_diamond_t0.nii.gz',
                    micro_path + 'diamond/' + subject + '_diamond_t1.nii.gz']

    peaks = np.stack((tensor_to_peak(nib.load(tensor_files[0]).get_fdata()),
                      tensor_to_peak(nib.load(tensor_files[1]).get_fdata())),
                     axis=4)

    fixel_weights = get_fixel_weight(trk, peaks)

    metric_list = ['FA', 'MD', 'RD', 'AD']

    if not os.path.isfile(micro_path + 'diamond/' + subject
                          + '_diamond_FA_t0.nii.gz'):

        create_tensor_metrics(micro_path + 'diamond/' + subject)

    for m in metric_list:

        map_files = [micro_path + 'diamond/' + subject + '_diamond_' + m
                     + '_t0.nii.gz', micro_path + 'diamond/' + subject
                     + '_diamond_' + m + '_t1.nii.gz']

        metric_maps = np.stack((nib.load(map_files[0]).get_fdata(),
                                nib.load(map_files[1]).get_fdata()),
                               axis=3)

        microstructure_map = get_microstructure_map(fixel_weights, metric_maps)
        mean, dev = get_weighted_mean(microstructure_map, fixel_weights)

        mean_dic[m] = mean
        dev_dic[m] = dev

    # Microstructure fingerprinting --------

    tensor_files = [micro_path + 'mf/' + subject + '_mf_peak_f0.nii.gz',
                    micro_path + 'mf/' + subject + '_mf_peak_f1.nii.gz']

    peaks = np.stack((nib.load(tensor_files[0]).get_fdata(),
                      nib.load(tensor_files[1]).get_fdata()),
                     axis=4)

    fixel_weights = get_fixel_weight(trk, peaks)

    metric_list = ['fvf', 'frac']

    for m in metric_list:

        map_files = [micro_path + 'mf/' + subject + '_mf_' + m + '_f0.nii.gz',
                     micro_path + 'mf/' + subject + '_mf_' + m + '_f1.nii.gz']

        metric_maps = np.stack((nib.load(map_files[0]).get_fdata(),
                                nib.load(map_files[1]).get_fdata()),
                               axis=3)

        microstructure_map = get_microstructure_map(fixel_weights, metric_maps)
        mean, dev = get_weighted_mean(microstructure_map, fixel_weights)

        mean_dic[m] = mean
        dev_dic[m] = dev

    return mean_dic, dev_dic


# %% Cell 9 - Analysis
def metrics_analysis(list_subjects: list, root: str, output_path: str, metric_name: list, edge_name: str):

    with open(edge_name, 'r') as read_file:
        edge_name = json.load(read_file)

    edge_name = add_regions(edge_name, ['cc_anterior_midbody_', 'cc_posterior_midbody_', 'cc_genu_', 'cc_splenium_', 'cc_isthmus_'])

    dataframe = {}

    dataframe['Mean'] = {}
    dataframe['Dev'] = {}

    for j, sub in enumerate(list_subjects):

        dataframe['Mean'][sub] = {}
        dataframe['Dev'][sub] = {}

        for i, r in enumerate(edge_name):

            if str(r[0]) not in ['cc_anterior_midbody_', 'cc_posterior_midbody_', 'cc_genu_', 'cc_splenium_', 'cc_isthmus_', 'uf_left', 'uf_right']:

                ROI = tract_to_ROI(root + '/subjects/' + sub
                                   + '/dMRI/tractography/tois/' + sub
                                   + '_tractogram_sift_' + str(r[1]) + '_'
                                   + str(r[0]) + '.trk')
                dataframe['Mean'][sub][str(r[1]) + '_' + str(r[0])] = {}
                dataframe['Dev'][sub][str(r[1]) + '_' + str(r[0])] = {}

            else:
                ROI = tract_to_ROI(root + '/subjects/' + sub
                                   + '/dMRI/tractography/tois/' + sub
                                   + '_tractogram_' + str(r[0]) + '.trk')

                dataframe['Mean'][sub][str(r[0])] = {}
                dataframe['Dev'][sub][str(r[0])] = {}

            for k, m in enumerate(metric_name):

                if m in ['FA', 'MD', 'RD', 'AD']:
                    model = 'dti'
                elif m in ['noddi_fintra', 'noddi_fextra', 'noddi_fiso',
                           'noddi_odi']:
                    model = 'noddi'
                elif m in ['diamond_wFA', 'diamond_wMD', 'diamond_wRD',
                           'diamond_wAD', 'diamond_frac_ftot',
                           'diamond_frac_csf']:
                    model = 'diamond'
                else:
                    model = 'mf'

                metric_map = nib.load(root + '/subjects/' + sub
                                      + '/dMRI/microstructure/' + model + '/'
                                      + sub + '_' + m + '.nii.gz').get_fdata()

                metric_in_ROI = metric_map[ROI != 0]

                mean_ROI = np.mean(metric_in_ROI[metric_in_ROI != 0])
                std_ROI = np.std(metric_in_ROI[metric_in_ROI != 0])

                if np.isnan(mean_ROI):
                    mean_ROI = 0
                if np.isnan(std_ROI):
                    std_ROI = 0

                if str(r[0]) not in ['cc_anterior_midbody_', 'cc_posterior_midbody_', 'cc_genu_', 'cc_splenium_', 'cc_isthmus_', 'uf_left', 'uf_right']:
                    dataframe['Mean'][sub][str(r[1]) + '_' + str(r[0])][m] = mean_ROI
                    dataframe['Dev'][sub][str(r[1]) + '_' + str(r[0])][m] = std_ROI
                else:
                    dataframe['Mean'][sub][str(r[0])][m] = mean_ROI
                    dataframe['Dev'][sub][str(r[0])][m] = std_ROI

    json.dump(dataframe, open(output_path + 'metric_analysis.json', 'w'),
              default=to_float64)

    return output_path + 'metric_analysis.json'


def get_mean_tracts_study(root: str, selected_edges_path: str,
                          output_path: str):
    '''


    Parameters
    ----------
    root : str
        DESCRIPTION.
    selected_edges_path : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    subjects_list = root + 'subjects/subj_list.json'

    with open(subjects_list, 'r') as read_file:
        subj_list = json.load(read_file)
    with open(selected_edges_path, 'r') as read_file:
        edge_list = json.load(read_file)

    subj_list = ['sub01_E1']

    dic_tot = {}
    dic_tot['Mean'] = {}
    dic_tot['Dev'] = {}

    for sub in subj_list:

        micro_path = root + 'subjects/' + sub + '/dMRI/microstructure/'
        tract_path = root + 'subjects/' + sub + '/dMRI/tractography/tois/'

        dic_tot['Mean'][sub] = {}
        dic_tot['Dev'][sub] = {}

        for edge in edge_list:

            try:
                trk_file = (tract_path + sub + '_tractogram_sift_'
                            + str(edge[0]) + '_' + str(edge[1]) + '.trk')

                mean_dic, dev_dic = get_mean_tracts(trk_file, micro_path)

            except FileNotFoundError:
                print('.trk file not found for edge ' + str(edge)
                      + ' in patient ' + sub)
                continue
            except IndexError:
                print('IndexError with subject ' + sub)
                continue

            dic_tot['Mean'][sub][str(edge)] = mean_dic
            dic_tot['Dev'][sub][str(edge)] = dev_dic

        toi_list = ['cc_genu_', 'cc_isthmus_', 'cc_posterior_midbody_',
                    'cc_anterior_midbody_', 'cc_splenium_', 'uf_left', 'uf_right']

        for toi in toi_list:

            try:

                trk_file = (tract_path + sub + '_tractogram_' + toi + '.trk')

                mean_dic, dev_dic = get_mean_tracts(trk_file, micro_path)
                print(tract_path + sub + '_tractogram_' + toi + '.trk')
            except FileNotFoundError:
                print('.trk file not found for edge ' + str(edge)
                      + ' in patient ' + sub)
                continue
            except IndexError:
                print('IndexError with subject ' + sub)
                continue

            dic_tot['Mean'][sub][toi] = mean_dic
            dic_tot['Dev'][sub][toi] = dev_dic

    json.dump(dic_tot, open(output_path + 'unravel_metric_analysis.json', 'w'),
              default=to_float64)
