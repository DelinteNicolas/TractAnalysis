import sys
from core import (register_atlas_to_subj, connectivity_matrices,
                  extract_streamline, get_edges_of_interest)

patient = sys.argv[1]
root = sys.argv[2]
code = sys.argv[3]

path_to_analysis_code = root.replace(
    root.split('/')[-2] + '/', '') + 'TractAnalysis/'

fa_path = (root + 'subjects/' + patient + '/dMRI/microstructure/dti/' + patient
           + '_FA.nii.gz')
atlas_path = path_to_analysis_code + 'data/atlas_desikan_killiany.nii.gz'
mni_fa_path = path_to_analysis_code + 'data/FSL_HCP1065_FA_1mm.nii.gz'
labels_path = (root + 'subjects/' + patient + '/masks/' + patient
               + '_labels.nii.gz')
dwi_path = (root + 'subjects/' + patient + '/dMRI/preproc/' + patient
            + '_dmri_preproc.nii.gz')
streamlines_path = (root + 'subjects/' + patient + '/dMRI/tractography/'
                    + patient + '_tractogram.trk')
matrix_path = root + 'subjects/' + patient + '/dMRI/tractography/' + patient
subjects_list = root + 'subjects/subj_list.json'
freeSurfer_labels = path_to_analysis_code + 'data/FreeSurfer_labels.xlsx'
output_path = path_to_analysis_code + 'output_analysis/'

if code == 'connectivity':

    register_atlas_to_subj(fa_path, atlas_path, mni_fa_path, labels_path)

    new_label_map = connectivity_matrices(
        dwi_path, labels_path, streamlines_path, matrix_path, freeSurfer_labels)

elif code == 'extraction':

    with open(output_path+'selected_edges.json') as file:
        edges = file.read()

    extract_streamline(edges[0], dwi_path, labels_path, streamlines_path)

else:

    print('Invalid code name')
