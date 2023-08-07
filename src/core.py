import os
import json
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from regis.core import find_transform, apply_transform
    from dipy.io.streamline import load_tractogram, save_trk
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.tracking import utils
from unravel.core import (get_fixel_weight, get_microstructure_map,
                          get_weighted_mean, tensor_to_peak)
from unravel.utils import tensor_to_DTI


def register_labels_to_atlas(labels_path: str, mni_fa_path: str,
                             output_path: str):
    '''


    Parameters
    ----------
    labels_path : str
        DESCRIPTION.
    mni_fa_path : str
        DESCRIPTION.
    output_path : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    map_desikan_to_fa = find_transform(labels_path, mni_fa_path,
                                       only_affine=True)

    apply_transform(labels_path, map_desikan_to_fa, static_file=mni_fa_path,
                    output_path=output_path, labels=True)


def register_atlas_to_subj(fa_path: str, label_path: str, mni_fa_path: str,
                           output_path: str, static_mask_path: str):
    '''
    Two-step registration to obtain label in the diffusion space.

    Parameters
    ----------
    fa_path : str
        Path to FA file.
    atlas_path : str
        DESCRIPTION.
    mni_fa_path : str
        DESCRIPTION.
    output_path : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    static_mask = nib.load(static_mask_path).get_fdata()

    map_mni_to_subj = find_transform(mni_fa_path, fa_path,
                                     hard_static_mask=static_mask)

    apply_transform(label_path, map_mni_to_subj, static_file=fa_path,
                    output_path=output_path, labels=True)


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

    unwanted = ['vessel', 'CSF', 'Vent', 'unknown', 'White_Matter', 'WM',
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
        0, len(labels_sorted) - 1, len(labels_sorted))
    new_label_map = np.zeros([len(labels), len(labels[0]), len(labels[0][0])])
    for i in range(len(labels_sorted)):
        new_label_map += np.where(labels
                                  == labels_sorted[i], new_labels_value[i], 0)
    new_label_map = new_label_map.astype('int64')

    M, _ = utils.connectivity_matrix(streams_data, affine,
                                     new_label_map,
                                     return_mapping=True,
                                     mapping_as_streamlines=True)

    np.fill_diagonal(M, 0)
    M = M.astype('float32')
    M = M / np.sum(M)

    fig, ax = plt.subplots()
    ax.imshow(np.log1p(M * len(trk.streamlines._offsets)),
              interpolation='nearest')
    ax.set_yticks(np.arange(len(area_sorted)))
    ax.set_yticklabels(area_sorted)

    trac = streamlines_path.replace('_tractogram', '_connectivity_matrix')
    trac_im = trac.replace('.trk', '.png')

    plt.savefig(trac_im)
    plt.title('Connectivity matrix')
    plt.xlabel("Labels")

    labels_path = streamlines_path.replace(
        '_tractogram', '_labels_connectivity_matrix')
    labels_path = labels_path.replace('.trk', '.txt')

    with open(labels_path, 'w') as f:
        for line in area_sorted:
            f.write(str(line) + '\n')

    trac_save = trac.replace('.trk', '.npy')
    np.save(trac_save, M)

    return new_label_map


def significance_level(list_subject: str, root: str, output_path: str):
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

    with open(list_subject, 'r') as read_file:
        subj_list = json.load(read_file)

    if os.path.isfile(list_subject.replace('subj_list.json', 'control_list.json')):

        with open(list_subject.replace('subj_list.json', 'control_list.json'), 'r') as control_file:
            control_list = json.load(control_file)

        copy_subj_list = []

        for i in range(len(subj_list)):
            if str(subj_list[i]) not in control_list:
                copy_subj_list.append(str(subj_list[i]))

        subj_list = copy_subj_list

    list_E1 = []
    list_E2 = []
    list_E3 = []

    for sub in subj_list:

        path = (root + 'subjects/' + str(sub)
                + '/dMRI/tractography/' + str(sub)
                + '_connectivity_matrix_sift.npy')
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

    # On part du principe que les entrées des matrices sont les mêmes, à vérifer
    pval_E12 = np.zeros((list_E1.shape[0], list_E1.shape[1]))
    pval_E13 = np.zeros((list_E1.shape[0], list_E1.shape[1]))
    pval_E23 = np.zeros((list_E1.shape[0], list_E1.shape[1]))

    for i in range(list_E1.shape[0]):
        for j in range(list_E1.shape[1]):
            _, pval_12 = ttest_ind(
                list_E1[i, j, :], list_E2[i, j, :], alternative='two-sided')
            if np.isnan(pval_12):
                pval_12 = 1
            pval_E12[i, j] = pval_12

            _, pval_13 = ttest_ind(
                list_E1[i, j, :], list_E3[i, j, :], alternative='two-sided')
            if np.isnan(pval_13):
                pval_13 = 1
            pval_E13[i, j] = pval_13

            _, pval_23 = ttest_ind(
                list_E2[i, j, :], list_E3[i, j, :], alternative='two-sided')
            if np.isnan(pval_23):
                pval_23 = 1
            pval_E23[i, j] = pval_23

    pval_all = np.stack([pval_E12, pval_E13, pval_E23], axis=2)

    np.save(output_path + 'pvals_E12_E13_E23.npy', pval_all)


def to_float64(val):
    """
    Used if *val* is an instance of numpy.float32.
    """

    return np.float64(val)


def get_edges_of_interest(pval_file: str, output_path: str,
                          min_path: str) -> list:
    '''
    Returns the edges corresponding to low p-values

    Parameters
    ----------
    pval_file : str
        Path to .npy containing p-values of connectivity matrices
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
    pval = np.transpose(pval, (2, 0, 1))
    pval = np.tril(pval, k=0)
    pval[pval == 0] = 1
    pval = np.transpose(pval, (1, 2, 0))

    # Removing regions where there are no connections
    mins = np.load(min_path)
    pval[mins == 0] = 1
    comparisons = np.count_nonzero(pval != 1) / pval.shape[2]

    # Multiple comparison pval correction
    # Bonferroni --------------------------------------

    pval_tresh = 0.05 / comparisons
    selec = np.argwhere(pval < pval_tresh)
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
        selec = np.argwhere(np.isin(pval, pval_cand_copy[0]))
        print('Minimum p-value used instead of multiple correction')

    # First value of candidate pvalues
    edge = tuple(selec[0][:2])

    json.dump([edge], open(output_path + 'selected_edges.json', 'w'),
              default=to_float64)


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

    streamlines = utils.target(streamlines, affine, mask1, include=True)
    streamlines = utils.target(streamlines, affine, mask2, include=True)

    tract = StatefulTractogram(streamlines, img, Space.RASMM)

    output_path = (streamlines_path.replace(streamlines_path.split('/')[-1], '')
                   + 'tois/')

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    filename = (output_path + streamlines_path.split('/')[-1][:-4] + '_'
                + str(edge[0]) + '_' + str(edge[1]))

    save_trk(tract, filename + '.trk')


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

        img = nib.load(path+'_diamond_'+tensor+'.nii.gz')
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
            out.to_filename(path+'_diamond_'+m+'_'+tensor+'.nii.gz')


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

    # Diamond ------------------------------

    tensor_files = [micro_path+'diamond/'+subject+'_diamond_t0.nii.gz',
                    micro_path+'diamond/'+subject+'_diamond_t1.nii.gz']

    tList = [tensor_to_peak(nib.load(tensor_files[0]).get_fdata()),
             tensor_to_peak(nib.load(tensor_files[1]).get_fdata())]

    fixel_weights, _, _ = get_fixel_weight(trk, tList, speed_up=True)

    metric_list = ['FA', 'MD', 'RD', 'AD']

    if not os.path.isfile(micro_path+'diamond/'+subject
                          + '_diamond_FA_t0.nii.gz'):

        create_tensor_metrics(micro_path+'diamond/'+subject)

    for m in metric_list:

        map_files = [micro_path+'diamond/'+subject+'_diamond_'+m+'_t0.nii.gz',
                     micro_path+'diamond/'+subject+'_diamond_'+m+'_t1.nii.gz']

        metricMapList = [nib.load(map_files[0]).get_fdata(),
                         nib.load(map_files[1]).get_fdata()]

        microstructure_map = get_microstructure_map(fixel_weights,
                                                    metricMapList)
        mean, dev = get_weighted_mean(microstructure_map, fixel_weights)

        mean_dic[m] = mean
        dev_dic[m] = dev

    # Microstructure fingerprinting --------

    tensor_files = [micro_path+'mf/'+subject+'_mf_peak_f0.nii.gz',
                    micro_path+'mf/'+subject+'_mf_peak_f1.nii.gz']

    tList = [nib.load(tensor_files[0]).get_fdata(),
             nib.load(tensor_files[1]).get_fdata()]

    fixel_weights, _, _ = get_fixel_weight(trk, tList, speed_up=True)

    metric_list = ['fvf', 'frac']

    for m in metric_list:

        map_files = [micro_path+'mf/'+subject+'_mf_'+m+'_f0.nii.gz',
                     micro_path+'mf/'+subject+'_mf_'+m+'_f1.nii.gz']

        metricMapList = [nib.load(map_files[0]).get_fdata(),
                         nib.load(map_files[1]).get_fdata()]

        microstructure_map = get_microstructure_map(fixel_weights,
                                                    metricMapList)
        mean, dev = get_weighted_mean(microstructure_map, fixel_weights)

        mean_dic[m] = mean
        dev_dic[m] = dev

    return mean_dic, dev_dic


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

    dic_tot = {}
    dic_tot['Mean'] = {}
    dic_tot['Dev'] = {}

    for sub in subj_list:

        micro_path = root+'subjects/'+sub+'/dMRI/microstructure/'
        tract_path = root+'subjects/'+sub+'/dMRI/tractography/tois/'

        dic_tot['Mean'][sub] = {}
        dic_tot['Dev'][sub] = {}

        for edge in edge_list:

            try:
                trk_file = (tract_path + sub + '_tractogram_sift_'
                            + str(edge[0]) + '_' + str(edge[1])+'.trk')

                mean_dic, dev_dic = get_mean_tracts(trk_file, micro_path)

            except FileNotFoundError:
                print('.trk file not found for edge ' + str(edge)+' in patient '
                      + sub)
                continue

            dic_tot['Mean'][sub][str(edge)] = mean_dic
            dic_tot['Dev'][sub][str(edge)] = dev_dic

    json.dump(dic_tot, open(output_path+'unravel_means.json', 'w'),
              default=to_float64)


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
        patient_list = json.load(open(root + 'subjects/subj_list.json', "r"))

    path_to_analysis_code = root.replace(
        root.split('/')[-2] + '/', '') + 'TractAnalysis/'
    path_to_code = path_to_analysis_code + 'src/scripts.py'

    for patient in patient_list:

        os.system('sbatch -J ' + patient + ' '
                  + path_to_analysis_code + 'slurm/submitIter.sh '
                  + path_to_code + ' ' + patient + ' ' + root + ' ' + code)
