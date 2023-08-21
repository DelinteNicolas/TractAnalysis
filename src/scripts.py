import sys
import json
from core import (register_atlas_to_subj, connectivity_matrices,
                  extract_streamline, get_mean_tracts_study)

# Arguments

patient = sys.argv[1]
root = sys.argv[2]
code = sys.argv[3]

# Pathways

path_to_analysis_code = (root.replace(root.split('/')[-2] + '/', '')
                         + 'TractAnalysis/')
fa_path = (root + 'subjects/' + patient + '/dMRI/microstructure/dti/' + patient
           + '_FA.nii.gz')
atlas_path = path_to_analysis_code + 'data/atlas_desikan_killiany.nii.gz'
mni_fa_path = path_to_analysis_code + 'data/FSL_HCP1065_FA_1mm.nii.gz'
labels_path = (root + 'subjects/' + patient + '/masks/' + patient
               + '_labels.nii.gz')
output_analysis_path = path_to_analysis_code + 'output_analysis/'
label_atlas_path = output_analysis_path+'atlas_desikan_killiany_mni.nii.gz'
dwi_path = (root + 'subjects/' + patient + '/dMRI/preproc/' + patient
            + '_dmri_preproc.nii.gz')
static_mask_path = (root + 'subjects/' + patient + '/masks/' + patient
                    + '_brain_mask.nii.gz')
# streamlines_path = (root + 'subjects/' + patient + '/dMRI/tractography/'
#                    + patient + '_tractogram.trk')
streamlines_path = (root + 'subjects/' + patient + '/dMRI/tractography/'
                    + patient + '_tractogram_sift.trk')
subjects_list = root + 'subjects/subj_list.json'
freeSurfer_labels = path_to_analysis_code + 'data/FreeSurfer_labels.xlsx'
excel_path = freeSurfer_labels.replace('.', '_bis.')
selected_edges_path = output_analysis_path+'selected_edges.json'

# Scripts and functions

if code == 'connectivity':

    register_atlas_to_subj(fa_path, label_atlas_path, mni_fa_path, labels_path,
                           static_mask_path=static_mask_path)

    new_label_map = connectivity_matrices(dwi_path, labels_path,
                                          streamlines_path,
                                          output_analysis_path,
                                          freeSurfer_labels)

elif code == 'extraction':

    with open(output_analysis_path + 'selected_edges.json', "r") as file:
        edges = json.load(file)

    for edge in edges:
        extract_streamline(edge, labels_path, streamlines_path, excel_path)

elif code == 'estimation':

    get_mean_tracts_study(root, selected_edges_path, output_analysis_path)

else:

    print('Invalid code name')
