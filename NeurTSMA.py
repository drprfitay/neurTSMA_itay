import internal
import numpy as np

from visualize_datasets import visualize_dataset

def initialize_environment(working_directory):
    """
    Initializes the environment if not already initialized.

    Parameters:
    - working_directory (str): The directory to set as the working directory.

    Returns:
    None
    """
    internal._initialize_environment(working_directory)
    

def create_dataset(matrix=None, dataset_name=None):
    """
    Creates a dataset.
    
    Parameters:
    - matrix (numpy.ndarray): The data matrix for the dataset.
    - dataset_name (str): The name of the dataset.
    
    Returns:
    Created dataset
    """

    return(internal._create_dataset(matrix, dataset_name))



def process_dataset(dataset_name=None, alias=None, prefix=None, op_list=list()):
    """
    Processes a dataset.
    
    Parameters:
    - dataset_name (str): The name of the dataset.
    - alias (str, optional): An alias for the dataset.
    - prefix (str, optional): A prefix for the dataset.
    - op_list (list): A list of operations to perform.
    
    Returns:
    Processed dataset
    """

    return(internal._process_dataset(dataset_name=dataset_name,
                                     alias=alias,
                                     prefix=prefix,
                                     op_list=op_list))


def get_dataset(dataset_name=None, alias=None, prefix=None):
    """
    Fetches a dataset. If alias and prefix are not set, fetches
    originally saved dataset

    Parameters:
    - dataset_name (str): The name of the dataset .
    - alias (str, optional): An alias for the dataset.
    - prefix (str, optional): A prefix for the dataset.

    Returns:
    Fetched dataset
    """
    
    return(internal._get_dataset(dataset_name=dataset_name,
                                 alias=alias,
                                 prefix=prefix))


def estimate_dimensionality(matrix=None, dataset_name=None, alias=None, prefix=None,
							exclusion_fraction_dimensionality_estimation=0.04):
    """
    Estimates the dimensionality of a dataset. If dataset_name is not set, estimates
    dimensionality over `matrix`

    Parameters:
    - matrix (numpy.ndarray, optional): The data matrix for the dataset.
    - dataset_name (str, optional): The name of the dataset.
    - alias (str, optional): An alias for the dataset.
    - prefix (str, optional): A prefix for the dataset.
    - exclusion_fraction_dimensionality_estimation (float, optional): Fraction of datapoints to exclude.

    Returns:
    float: Estimated dimensionality of the dataset.
    """    
    
    if matrix is None:
        matrix = get_dataset(dataset_name=dataset_name,
                             alias=alias,
                             prefix=prefix)

    # Also kinda silly
    return(internal._get_intrinsic_dimension(matrix,
                                             percentile=(1-exclusion_fraction_dimensionality_estimation)))
    

def cluster_dataset(dataset_name=None, 
                    alias=None, 
                    prefix=None,
                    number_of_clusters=None,
                    pointcloud_size=None,
                    timepoints_to_include=None,
                    kmeans_reps=500):
    """
    Clusters a dataset. If `pointcloud_size` is set ignores `timepoints_to_include`

    Parameters:
    - dataset_name (str): The name of the dataset.
    - alias (str, optional): An alias for the dataset.
    - prefix (str, optional): A prefix for the dataset.
    - number_of_clusters (int, optional): The number of clusters to form.
    - pointcloud_size (float, optional): Fraction of datapoints to use for clustering, between [0,1].
    - timepoints_to_include (int, optional): Number of timepoints to include in clustering.
    - kmeans_reps (int, optional): Number of repetitions for k-means.

    Returns:
    Clustered dataset labels
    """
    
    # This is kinda shitty, this logic should be exported elsewhere
    if pointcloud_size is not None and pointcloud_size >= 0 and pointcloud_size <= 1:
        matrix = get_dataset(dataset_name=dataset_name, alias=alias, prefix=prefix)
        timepoints_to_include = int(np.floor(matrix.shape[1] * (1-pointcloud_size)))
        
    
    return(internal._stochastic_kmeans_clustering(dataset_name=dataset_name,
                                                  alias=alias,
                                                  prefix=prefix,
                                                  number_of_clusters=number_of_clusters,
                                                  timepoints_to_include=timepoints_to_include,
                                                  stochastic_kmeans_reps=kmeans_reps))


def get_cluster_map(dataset_name=None, 
                    alias=None,
                    prefix=None,
                    max_cluster_number=20, 
                    min_num_of_frames=500,
                    timepoint_bins=40,
                    map_reps=20,
                    kmeans_reps=500):
    """
    Builds a cluster map unless already exists.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - alias (str, optional): An alias for the dataset.
    - prefix (str, optional): A prefix for the dataset.
    - max_cluster_number (int, optional): Maximum number of clusters.
    - min_num_of_frames (int, optional): Minimum number of frames per cluster.
    - timepoint_bins (int, optional): Number of timepoint bins.
    - map_reps (int, optional): Number of repetitions for cluster mapping.
    - kmeans_reps (int, optional): Number of repetitions for k-means.

    Returns:
    A dictionary containing optimal number of clusters, excluded datapoints, clusters heatmap
    and labels for clustering under optimal configuration.
    """
    
    return(internal._process_cluster_map(dataset_name=dataset_name,
                                         alias=alias,
                                         prefix=prefix,
                                         max_cluster_number=max_cluster_number,
                                         min_num_of_frames=min_num_of_frames,
                                         timepoint_bins=timepoint_bins,
                                         nreps=map_reps,
                                         stochastic_kmeans_reps=kmeans_reps))



# ToDO implement this functionality
def calculate_mutual_information_with_clustered_dataset(dataset_name=None, 
                                                        alias=None,
                                                        prefix=None,
                                                        variable_labels=None):


#     Parameters:
#     - matrix (numpy.ndarray, optional): The data matrix for the dataset.
#     - dataset_name (str, optional): The name of the dataset.
#     - preset_name (str, optional): The name of the preset to use.
#     - clustered_labels (numpy.ndarray, optional): The labels obtained from clustering.
#     - variable_labels (list, optional): The labels for variables.

#     Returns:
#     None
#     """
    
     pass

base_environment = "C:\\Users\\itayta\\Documents\\NeurTSMA"
initialize_environment(base_environment)
#create_dataset(`Enter your matrix`, "IC44_170518")



##### Example 1:
bin_params = {'bin_size':15}
scale_params = {'scale':"Zscore"}
filter_params = {'threshold':.5}
lem_iter_1_params = {'method': "lem", 'ndim': 20, 'knn':.075}
lem_iter_2_params = {'method': "lem", 'ndim': 6, 'knn':.025}

dataset_processing = [{'opcode': internal.BIN,    'params':  bin_params},\
                      {'opcode': internal.FILTER, 'params':  filter_params},\
                      {'opcode': internal.SCALE,  'params':  scale_params,      'alias':"high_dim_clean"},\
                      {'opcode': internal.REDUCE, 'params':  lem_iter_1_params, 'alias':"lem_1"},\
                      {'opcode': internal.REDUCE, 'params':  lem_iter_2_params, 'alias':"lem_final"}]
    
    
process_dataset("IC44_170518", op_list=dataset_processing)
reduced_dataset = get_dataset("IC44_170518", alias="lem_final")
cluster_map = get_cluster_map(dataset_name="IC44_170518", alias="lem_final")

lem_1 = get_dataset("IC44_170518", alias="lem_1")
est_dim = estimate_dimensionality(matrix=lem_1)

color_map = {-1:"#5E4FA2",1:"#48A0B2",\
              2:"#A1D9A4",3:"#EDF7A3",\
              4:"#FEE899",5:"#FBA45C",\
              6:"#E25249",7:"#9E0142"}



visualize_dataset(reduced_dataset)
    
visualize_dataset(reduced_dataset,
                  variable=cluster_map["cluster_labels"],
                  color_map=color_map)


    
##### Example 2:
TSNE_processing = [{'opcode': internal.REDUCE, 
                    'params':  {'method':'tsne', 'ndim':3, 'perplexity':50, 'n_iter':800}, 
                    'alias':"tsne_example"}]

reduced_tsnse = process_dataset("IC44_170518", 
                                alias="high_dim_clean", 
                                op_list=TSNE_processing)



##### Example 3:
MDS_processing = [{'opcode': internal.REDUCE, 
                    'params':  {'method':'mds', 'ndim':6}, 
                    'alias':"mds_example"}]

reduced_mds = process_dataset("IC44_170518", 
                              alias="high_dim_clean", 
                              op_list=MDS_processing)



##### Example 4:
isomap_pca_processing = [{'opcode': internal.BIN,    'params':  bin_params},\
                         {'opcode': internal.FILTER, 'params':  filter_params},\
                         {'opcode': internal.SCALE,  'params':  scale_params},\
                         {'opcode': internal.REDUCE, 'params':  {'method':'pca', 'ndim':100}, 'alias':"pca_100"},\
                         {'opcode': internal.REDUCE, 'params':  {'method':'isomap', 'ndim':6, 'n_neighbors':70}, 'alias':"isomap_example"}]

reduced_iso_pca = process_dataset("IC44_170518", 
                                  op_list=isomap_pca_processing)

clustered_iso = cluster_dataset("IC44_170518", alias="isomap_example", number_of_clusters=7, pointcloud_size=.58)

est_dim_2 = estimate_dimensionality(dataset_name="IC44_170518", alias="pca_100")




