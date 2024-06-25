import matplotlib.pyplot as plt
import numpy as np

import itertools
import internal


color_map = {-1:"#5E4FA2",1:"#48A0B2",\
              2:"#A1D9A4",3:"#EDF7A3",\
              4:"#FEE899",5:"#FBA45C",\
              6:"#E25249",7:"#9E0142"}


dim_map = {2:[1,2],
           3:[1,3],
           4:[3,2],
           5:[5,2],
           6:[5,3],
           7:[7,3],
           8:[7,4],
           9:[9,4],
           10:[6,5]}


def _get_color_variable(variable, color_map):
    if not all([type(x) is int for x  in color_map.keys()]):
        internal._raise_error("Color map dictionary keys must be integers")
    
    keys = np.array([x for x in color_map.keys()])    
    keys = keys - np.min(keys)
    
    corrected_var = variable - min(variable)
    corrected_var = np.array(corrected_var)
    
    final_color_map = np.repeat("#606060", np.max(keys) + 1)
    values = np.array([color_map[x] for x in color_map.keys()])
    
    final_color_map[keys] = values
    
    return(final_color_map[corrected_var])

def visualize_dataset(mat, variable=None, color_map=None):
    
    if variable is not None and color_map is not None:
        color = _get_color_variable(variable, color_map)
    else:
        color = "#606060"
    
    dim = mat.shape[0]    
    
    num_rows = dim_map[dim][0]
    num_cols = dim_map[dim][1]
        
    all_planes = [x for x in itertools.combinations(range(0,dim), 2)]

    # Create a figure and axis objects
    fig, axs = plt.subplots(num_rows,  num_cols, figsize=(12, 15))

    # Loop through each subplot and set some properties
    plane_counter = 0
    for i in range(num_rows):
        for j in range(num_cols):
            dim1 = all_planes[plane_counter][0]
            dim2 = all_planes[plane_counter][1]
            plane_counter += 1
            
            ax = axs[i, j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.scatter(mat[dim1,:], mat[dim2,:], alpha=.35, color=color)  # Example plot
            ax.set_xlabel('Dim %d' % dim1)
            ax.set_ylabel('Dim %d' % dim2)
        
    plt.tight_layout()    
    plt.show()


#ToDo: clean this up
#def _plot_dataset():
#    colnames = [str(x) for x in range(0,6)] + ["color"]
#    fdf = pd.DataFrame(np.transpose(np.vstack([mat, [int(l) for l in final_labels]])), columns=colnames)
#    sns.pairplot(fdf, hue="color", palette=color_map)