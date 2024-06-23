import os 
import sys
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#figure out how to import it with local file reference. 
from .plotly_utils_test import compute_periodic_distance


def make_NNN(point_df, frame_cell_parameter_matrix, size = 100, distance="Euclidian"):
    """
    Creates a nearest neighbor networkfor 1000 randomly selected points from this array.
    We output a graph object: a tuple of nodes as ids, edges: describing connectivity,
    and node features. Ideally, we would ues the networkx package, but that is not working. 
    """

    point_array = point_df[["x", "y", "z"]].values
    point_indices = point_df.index.values

    frame_dims_perpendicular = frame_cell_parameter_matrix.diagonal()

    #Select 100 points randomly from points. 
    selection_mask = np.random.choice(point_array.shape[0], size=size, replace=False)

    coordinates = point_array[selection_mask, :]
    ids = np.arange(coordinates.shape[0])
    
    coordinates_compare, ids_compare = coordinates, ids

    edges = []
    for roll_id in np.arange(1, coordinates.shape[0], 1):
        #computing periodic distances. 
        ids_compare = np.roll(ids_compare, -1, axis=0)
        coordinates_compare = np.roll(coordinates_compare, roll_id, axis=0)
        distances = compute_periodic_distance(coordinates, coordinates_compare,
             frame_dims_perpendicular)
        cutoff_distance = 6 #A #select one to ensure that is larger than the cluster-cluster distance. 
        mask = np.logical_and(distances > 4, distances <  cutoff_distance)
        distances = distances.round(2)
        new_edges = [set([i, j]) for i, j in zip(ids[mask], ids_compare[mask])]
        edges += new_edges

    nodes = ids
    node_info = pd.DataFrame(point_array[selection_mask], columns=["x", "y", "z"])
    for column in point_df.columns:
        if column not in ["x", "y", "z"]:
            node_info[column] = point_df.loc[point_indices[selection_mask], column].values
    node_info["og_index"] = point_indices[selection_mask]

    return (nodes, edges, node_info)


def plot_network_traces(NNN, fig=None):
    """
    Plot network traces given the network. 
    Here, we actually add network traces to the figure. There are two traces
    which we can add: the network lines trace and network points trace.  
    """
    nodes , edges, node_info = NNN
    #We will now plot these on the figure. This can also be made into a function and written as a utility. 
    if fig is None:
        fig = go.Figure()

    network_trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines', \
                            line=dict(width = 10, color="black"), visible=True,
                            name="network_trace")
    for (i, j) in edges:
        cart_distance_array  = node_info.loc[i, ["x", "y", "z"]].values - node_info.loc[j, ["x", "y", "z"]].values
        cart_distance = np.sqrt(np.sum(cart_distance_array**2))
        if cart_distance < 4 :
            network_trace['x'] += (node_info.loc[i, "x"], node_info.loc[j, "x"], None)
            network_trace['y'] += (node_info.loc[i, "y"], node_info.loc[j, "y"], None)
            network_trace['z'] += (node_info.loc[i, "z"], node_info.loc[j, "z"], None)

    #We add another trace to show the points. 
    network_points_trace = go.Scatter3d(x=node_info["x"].values, y=node_info["y"].values, z=node_info["z"].values, \
            mode='markers', marker=dict(color='black', size=7, opacity=1),
            name='network_points_trace', visible=True)
    
    fig.add_traces([network_trace, network_points_trace])

#%%

# %%
