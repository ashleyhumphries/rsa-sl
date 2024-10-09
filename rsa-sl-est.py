
#import other scripts
from functions import upper_tri, generate_null_distribution, fisher_z_transform
from test_functions import check_bold_data_shape, check_mask_shape, check_data_reordering, check_RDM_len, test_scores_are_not_zeros
# Standard libraries
import os.path as op
import re

# Scientific computing libraries
import numpy as np
import pandas as pd

# Neuroimaging libraries
import nibabel as nib
from nilearn import image

# Machine learning libraries
from sklearn.metrics import pairwise_distances



# File handling
from glob import glob

# RSA toolbox libraries
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
import rsatoolbox as rsatoolbox
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight

# Set printing precision
np.set_printoptions(precision=2, suppress=True)




#sub = ['sub-4', 'sub-5','sub-6','sub-7','sub-8', 'sub-9', 'sub-10', 'sub-11', 'sub-12' ,'sub-13','sub-14', 'sub-15', 'sub-16', \
#'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25' ,'sub-26', 'sub-27', 'sub-28', 'sub-29',
#'sub-30', 'sub-31', 'sub-32', 'sub-33', 'sub-34', 'sub-35'] 


sub = ['sub-4', 'sub-5','sub-6','sub-7','sub-8', 'sub-9', 'sub-11', 'sub-12' ,'sub-13','sub-14', 'sub-15', 'sub-16', \
'sub-17', 'sub-18','sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25' ,'sub-27','sub-29',
'sub-30', 'sub-31', 'sub-32', 'sub-33', 'sub-34', 'sub-35'] 




for subject in sub:
    derivatives_dir = op.join(f'/work/cb3/ahumphries/derivatives/{subject}')
    func_dir = op.join(derivatives_dir, "ses-1/func/")
    bold_files = sorted(glob(op.join(func_dir, '*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')))

    bids_dir= op.join(f'/work/cb3/ahumphries/all-subs-bids v2/{subject}')
    events_dir = op.join(bids_dir + "/ses-1/func/")
    event_files = sorted(glob(op.join(events_dir, '*_events.tsv')))



    # Iterate over all BOLD files (runs 1-8)
    for run_num, bold_file in enumerate(bold_files, start=1):
        print(f"Processing BOLD file for run {run_num}: {bold_file}")
        
        # Load BOLD file
        bold_img = image.load_img(bold_file)
        bold_data = bold_img.get_fdata()

        # Load corresponding whole brain mask
        wb_mask_file = op.join(func_dir, f'{subject}_ses-1_task-VB_run-{run_num}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
        whole_brain_mask_img = nib.load(wb_mask_file)
        whole_brain_mask_data = whole_brain_mask_img.get_fdata()

        # Load corresponding binary brain mask (assuming it's the same as wb_mask_file)
        binary_mask_img = image.load_img(wb_mask_file)
        binary_mask_data = binary_mask_img.get_fdata()

        # Create mask from binary mask to use for SL (x, y, z)
        mask = binary_mask_data.astype(bool)  # Converts the 1s and 0s into True/False


        # Check BOLD data shape
        expected_bold_shape = (65, 77, 56, 180)
        check_bold_data_shape(bold_data, expected_bold_shape)
        # Check binary mask shape
        expected_mask_shape = (65, 77, 56)
        check_mask_shape(binary_mask_data, expected_mask_shape)

    
    #1b: find the SL centers! threshold is proportion of voxels that need to be inside mask to be considered a good SL center
        centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=0.5)

    #2a: Load event file
        run_pattern = re.compile(f'_run-{run_num}_')

        # Use regex search to find the correct file for the specified run
        event_file = next((path for path in event_files if run_pattern.search(path)), None)
        print(event_file)
        event = pd.read_csv(event_file, sep='\t')

        event_sans_fix = event[event['stimuli type'] != 'fixation']


    #2b:creating trial list

    # Define the list of desired stimuli types
        face_stimuli = ['Surprise', 'Happy', 'Angry']
        scene_stimuli = ['AMBIG', 'POS', 'NEG']
    # Filter the DataFrame for face and scene

        face = event_sans_fix[event_sans_fix['stimuli type'].isin(face_stimuli)]
        face = face[face['Task'] == 'Valence']
        scene = event_sans_fix[event_sans_fix['stimuli type'].isin(scene_stimuli)]
        scene = scene[scene['Task'] == 'Valence']
        face_trials = list(face['Trial'])
        scene_trials = list(scene['Trial'])



    #2c: reshape BOLD data so we have n_obs x n_voxels sliced by trials

        #dropping all the fixation volumes, leaving only the stimuli by slicing the 4th dimension with the trial_list
    
        bold_face_data = bold_data[...,face_trials] 
        bold_scene_data = bold_data[...,scene_trials] 
    # Rearrange the dimensions so that the time dimension (4th dimension) becomes the first dimension
        bold_face_data_reordered = np.moveaxis(bold_face_data, -1, 0)
        bold_scene_data_reordered = np.moveaxis(bold_scene_data, -1, 0)
    #checks for reordering
        expected_reordered_shape = (24, 65, 77, 56)
        check_data_reordering(bold_face_data_reordered, expected_reordered_shape)
        check_data_reordering(bold_scene_data_reordered, expected_reordered_shape)

    #reshaping data from being 4D to 2D
        data_2d_face = bold_face_data_reordered.reshape([bold_face_data_reordered.shape[0], -1])
        data_2d_face = np.nan_to_num(data_2d_face)
        #scene
        data_2d_scene= bold_scene_data_reordered.reshape([bold_scene_data_reordered.shape[0], -1])
        data_2d_scene = np.nan_to_num(data_2d_scene)



    #for having seperate trials: 
        face_events_array= np.array(face['trial_type'])
        scene_events_array= np.array(scene['trial_type'])

    #2d: Iterate over all the searchlight centers and calculates the RDM

        SL_RDM = get_searchlight_RDMs(data_2d_face, centers, neighbors, face_events_array, method='correlation') #might want to be euclidean
        #SL_RDM1 = get_searchlight_RDMs(data_2d, centers, neighbors, events_array, method='euclidean') #eucliudean approach to calculating SLs
        SL_RDM1 = get_searchlight_RDMs(data_2d_scene, centers, neighbors, scene_events_array, method='correlation') #might want to be euclidean

    #3: Creating model RDMS (null RDMs and distribution created in other script)

    #pulling stimuli and mapping the labels
    #3a: face
        S1= face['stimuli type']
        neg_mapping = {'Angry': 1,'Surprise': 1, 'Happy': 0}
        pos_mapping = {'Angry': 0,'Surprise': 1, 'Happy': 1}

    #use list comprehension to encode the labels and make RDMs
        encoded_labels = [neg_mapping[label] for label in S1]
        neg_array =np.array(encoded_labels)
        encoded_labels = [pos_mapping[label] for label in S1]
        pos_array =np.array(encoded_labels)
        expected_RDM_len= (24)
        check_RDM_len(neg_array, expected_RDM_len)
        check_RDM_len(pos_array, expected_RDM_len)

        neg_f_vb_rdm = pairwise_distances(neg_array[:, np.newaxis], metric='manhattan')
        pos_f_vb_rdm = pairwise_distances(pos_array[:, np.newaxis], metric='manhattan')
        
    #3b: scene
        S1= scene['stimuli type']
        neg_mapping = {'AMBIG': 1,'POS':0,'NEG':1}
        pos_mapping = {'AMBIG': 1,'POS':1,'NEG':0}

    #use list comprehension to encode the labels and make RDMs
        encoded_labels = [neg_mapping[label] for label in S1]
        neg_array =np.array(encoded_labels)
        encoded_labels = [pos_mapping[label] for label in S1]
        pos_array =np.array(encoded_labels)


        neg_s_vb_rdm = pairwise_distances(neg_array[:, np.newaxis], metric='manhattan')
        pos_s_vb_rdm = pairwise_distances(pos_array[:, np.newaxis], metric='manhattan')
    #making the face /scenemodels 
        neg_vb_model = ModelFixed('Neg VB RDM', upper_tri(neg_f_vb_rdm))
        pos_vb_model = ModelFixed('Pos VB RDM', upper_tri(pos_f_vb_rdm))

        negsce_vb_model = ModelFixed('Neg VB RDM', upper_tri(neg_s_vb_rdm))
        possce_vb_model = ModelFixed('Pos VB RDM', upper_tri(pos_s_vb_rdm))

    

    #4: evaluating the models 

    #4a: face

        models = [pos_vb_model, neg_vb_model] # shuff_vb_model

    # Evaluate the models using evaluate_models_searchlight
        eval_results = evaluate_models_searchlight(SL_RDM, models, eval_fixed, method='spearman', n_jobs=16)

    # Unpack the evaluation scores
        eval_scores = [[float(e.evaluations[0, i, 0]) for e in eval_results] for i in range(len(models))]
    #saving the correlations for each voxel SL for later
        pos_face_corrs = eval_scores[0]
        neg_face_corrs = eval_scores[1]

        average_scores = [np.mean(scores) for scores in eval_scores]
    #average_noise_ceiling = np.mean(noise_ceilings)

    # Determine the best model
        best_model_index = np.argmax(average_scores)
        best_model = models[best_model_index]

        print(f"The best performing model for face run {run_num} is: Model {best_model_index + 1}")
        print(f"Average scores:", average_scores)
    #4b: scene
        models = [possce_vb_model, negsce_vb_model]

    # Evaluate the models using evaluate_models_searchlight
        eval_results = evaluate_models_searchlight(SL_RDM1, models, eval_fixed, method='spearman', n_jobs=16)

    # Unpack the evaluation scores and noise ceilings
        eval_scores = [[float(e.evaluations[0, i, 0]) for e in eval_results] for i in range(len(models))]

    #saving the correlations for each voxel SL for later
        pos_scene_corrs = eval_scores[0]
        neg_scene_corrs = eval_scores[1]

        average_scores = [np.mean(scores) for scores in eval_scores]

    # Determine the best model
        best_model_index = np.argmax(average_scores)
        best_model = models[best_model_index]


        print(f"The best performing model for scene run {run_num} is: Model {best_model_index + 1}")
        print(f"Average scores:", average_scores)



    # 5: Null distribution creating
    # see function in the functions.py file for more details 
    # should take around ~3 hrs / run 
        # nperms=1000

        # print("Permutation testing start.")

        # face_scores, scene_scores = generate_null_distribution(SL_RDM, SL_RDM1, centers, nperms)

        # #print(face_scores[:100,1])
        # #print(scene_scores[:100,1])
        # print("Permutation  testing done.")
        # test_scores_are_not_zeros(face_scores, scene_scores)


        # # at each voxel, add up how many nulls were as or more extreme to the initial correlation
        # # divided by the number of nulls we have (nperms)
        # # anything below 0.05 is sig. 
        # # inverted so anything above 0.95 (95%) is sig. 
        # p_pos_face = np.zeros(centers.shape[0])
        # for i in range(centers.shape[0]):
        #     p_pos_face[i] = 1 - (np.sum(face_scores[i,:] >= pos_face_corrs[i]) / nperms)

        # p_neg_face = np.zeros(centers.shape[0])
        # for i in range(centers.shape[0]):
        #     p_neg_face[i] = 1 - (np.sum(face_scores[i,:] >= neg_face_corrs[i]) / nperms)

        # p_pos_scene = np.zeros(centers.shape[0])
        # for i in range(centers.shape[0]):
        #     p_pos_scene[i] = 1 - (np.sum(scene_scores[i,:] >= pos_scene_corrs[i]) / nperms)

        # p_neg_scene = np.zeros(centers.shape[0])
        # for i in range(centers.shape[0]):
        #     p_neg_scene[i] = 1 - (np.sum(scene_scores[i,:] >= neg_scene_corrs[i]) / nperms)

    
     #6: Plotting SL Maps
     #creates a 3d array of zeros the same shape as the mask, and then assigns the p-val array to the 3d array, based on 
     #the list of the voxel index. so the p-vals will display in the correct SL center
     #saves it to a nifti file
     
        x, y, z = mask.shape

        # #POS 
        # pos_pval_brain = np.zeros([x*y*z])
        # pos_pval_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = p_pos_face
        # pos_pval_brain = pos_pval_brain.reshape([x,y,z])
        # pos_face_p_brain_img = nib.Nifti1Image(pos_pval_brain, affine=bold_img.affine, header=bold_img.header)

        # #NEG
        # neg_pval_brain = np.zeros([x*y*z])
        # neg_pval_brain[list(SL_RDM.rdm_descriptors['voxel_index'])] = p_neg_face
        # neg_pval_brain = neg_pval_brain.reshape([x,y,z])
        # neg_face_p_brain_img = nib.Nifti1Image(neg_pval_brain, affine=bold_img.affine, header=bold_img.header)

        # #POS
        # pos_pval_brain_s = np.zeros([x*y*z])
        # pos_pval_brain_s[list(SL_RDM.rdm_descriptors['voxel_index'])] = p_pos_scene
        # pos_pval_brain_s = pos_pval_brain_s.reshape([x,y,z])
        # pos_scene_p_brain_img = nib.Nifti1Image(pos_pval_brain_s, affine=bold_img.affine, header=bold_img.header)

        # #NEG
        # neg_pval_brain_s = np.zeros([x*y*z])
        # neg_pval_brain_s[list(SL_RDM.rdm_descriptors['voxel_index'])] = p_neg_scene
        # neg_pval_brain_s = neg_pval_brain_s.reshape([x,y,z])
        # neg_scene_p_brain_img = nib.Nifti1Image(neg_pval_brain_s, affine=bold_img.affine, header=bold_img.header)

        output_dir = "/work/cb3/ahumphries/RSA-SL/maps"

        #nifti file outputs for each sub and run 
        #FACE
        # nib.save(pos_face_p_brain_img, op.join(output_dir,f'POSf_map_p-run_{run_num}-{subject}.nii.gz'))
        # nib.save(neg_face_p_brain_img, op.join(output_dir,f'NEGf_map_p-run_{run_num}-{subject}.nii.gz'))
        # #SCENE
        # nib.save(pos_scene_p_brain_img, op.join(output_dir,f'POSs_map_p-run_{run_num}-{subject}.nii.gz'))
        # nib.save(neg_scene_p_brain_img, op.join(output_dir,f'NEGs_map_p-run_{run_num}-{subject}.nii.gz'))


     #5a: Create RDM brain maps by reshaping the 3d arrays
        #POS
        pos_corrs_f = np.zeros([x*y*z])
        pos_corrs_f[list(SL_RDM.rdm_descriptors['voxel_index'])] = pos_face_corrs
        pos_corrs_f= pos_corrs_f.reshape([x,y,z])

        pos_corrs_f_z = fisher_z_transform(pos_corrs_f)

        pos_corrs_s = np.zeros([x*y*z])
        pos_corrs_s[list(SL_RDM1.rdm_descriptors['voxel_index'])] = pos_scene_corrs
        pos_corrs_s = pos_corrs_s.reshape([x,y,z])

        pos_corrs_s_z = fisher_z_transform(pos_corrs_s)
        #NEG
        neg_corrs_f = np.zeros([x*y*z])
        neg_corrs_f[list(SL_RDM.rdm_descriptors['voxel_index'])] = neg_face_corrs
        neg_corrs_f = neg_corrs_f.reshape([x,y,z])

        neg_corrs_f_z = fisher_z_transform(neg_corrs_f)

        neg_corrs_s = np.zeros([x*y*z])
        neg_corrs_s[list(SL_RDM1.rdm_descriptors['voxel_index'])] = neg_scene_corrs
        neg_corrs_s = neg_corrs_s.reshape([x,y,z])

        neg_corrs_s_z = fisher_z_transform(neg_corrs_s)

        # Create a Nifti image from the correlation array using the bold affine
        pos_face_corrs_brain_img = nib.Nifti1Image(pos_corrs_f, affine=bold_img.affine, header=bold_img.header)
        neg_face_corrs_brain_img = nib.Nifti1Image(neg_corrs_f, affine=bold_img.affine, header=bold_img.header)

        pos_face_corrs_brain_img_z = nib.Nifti1Image(pos_corrs_f_z, affine=bold_img.affine, header=bold_img.header)
        neg_face_corrs_brain_img_z = nib.Nifti1Image(neg_corrs_f_z, affine=bold_img.affine, header=bold_img.header)

        pos_scene_corrs_brain_img = nib.Nifti1Image(pos_corrs_s, affine=bold_img.affine, header=bold_img.header)
        neg_scene_corrs_brain_img = nib.Nifti1Image(neg_corrs_s, affine=bold_img.affine, header=bold_img.header)

        pos_scene_corrs_brain_img_z = nib.Nifti1Image(pos_corrs_s_z, affine=bold_img.affine, header=bold_img.header)
        neg_scene_corrs_brain_img_z = nib.Nifti1Image(neg_corrs_s_z, affine=bold_img.affine, header=bold_img.header)

    #     # Save the NIfTI images to a file.
        nib.save(pos_face_corrs_brain_img,  op.join(output_dir,f'POSf_corr_map_{run_num}-{subject}.nii.gz'))
        nib.save(neg_face_corrs_brain_img,  op.join(output_dir,f'NEGf_corr_map_{run_num}-{subject}.nii.gz'))

        nib.save(pos_face_corrs_brain_img_z,  op.join(output_dir,f'POSf_corr_map-z-scored_{run_num}-{subject}.nii.gz'))
        nib.save(neg_face_corrs_brain_img_z,  op.join(output_dir,f'NEGf_corr_map-z-scored_{run_num}-{subject}.nii.gz'))

        nib.save(pos_scene_corrs_brain_img,  op.join(output_dir,f'POSs_corr_map_{run_num}-{subject}.nii.gz'))
        nib.save(neg_scene_corrs_brain_img,  op.join(output_dir,f'NEGs_corr_map_{run_num}-{subject}.nii.gz'))

        nib.save(pos_scene_corrs_brain_img_z,  op.join(output_dir,f'POSs_corr_map-z-scored_{run_num}-{subject}.nii.gz'))
        nib.save(neg_scene_corrs_brain_img_z,  op.join(output_dir,f'NEGs_corr_map-z-scored_{run_num}-{subject}.nii.gz'))

        print(f"Finished processing run {run_num}\n")


print(f"Finished running {sub}\n")


if __name__ == "__main__":
    
    pass