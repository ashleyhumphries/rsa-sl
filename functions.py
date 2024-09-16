
#imports for functions 

import matplotlib.colors
import numpy as np
from sklearn.metrics import pairwise_distances
# RSA toolbox libraries
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
import rsatoolbox as rsatoolbox
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight





def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]



def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap


def generate_null_distribution(SL_RDM, SL_RDM1, centers, nperms=1000):
    """
    Generate null distributions for face and scene scores using permutation testing.

    Parameters:
    - SL_RDM: The searchlight RDM for the face model.
    - SL_RDM1: The searchlight RDM for the scene model.
    - nperms: Number of permutations to run (default is 1000).
    Returns:
    - face_scores: The generated null distribution for face scores.
    - scene_scores: The generated null distribution for scene scores.
    """

    face_scores = np.zeros((centers.shape[0],nperms))
    scene_scores = np.zeros((centers.shape[0],nperms))
    array = np.array([1]*18 + [0]*6)
    
    for i in range(nperms):
            
        shuffled_arr = np.random.permutation(array)
    #print("Shape of Shuffled VB RDM:", shuffled_arr.shape) #would expect this to be the same shape as bold_data in t dimension

        shuff_rdm = pairwise_distances(shuffled_arr[:, np.newaxis], metric='manhattan')
  
    #making the face / scene model 
        shuff_vb_model = ModelFixed('Shuf VB RDM', upper_tri(shuff_rdm))


        eval_results = evaluate_models_searchlight(SL_RDM,shuff_vb_model,eval_fixed, method = 'spearman', n_jobs= 16) #spearman way
        eval_score = [float(e.evaluations) for e in eval_results]
        face_scores[:, i] = eval_score   

        eval_results = evaluate_models_searchlight(SL_RDM1,shuff_vb_model,eval_fixed, method = 'spearman', n_jobs= 16) #spearman way
        eval_score = [float(e.evaluations) for e in eval_results]
        scene_scores[:, i] = eval_score

    return face_scores, scene_scores


# Define a function to apply Fisher's Z-transformation
def fisher_z_transform(corr_map):
    # Clip the correlation values to avoid any -1 or 1 values
    corr_map = np.clip(corr_map, -0.999999, 0.999999)
    # Apply Fisher's Z-transformation
    return np.arctanh(corr_map)