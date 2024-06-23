"""
Implements functions from the AdsVis library.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from scipy.interpolate import griddata as gd
import plotly.graph_objects as go

from src import BASE_DIR

from .plotly_utils_test import make_cell_parameter_matrix, compute_RDF
from .network_utils import make_NNN, plot_network_traces


DATAPATH = BASE_DIR / 'Demos/data'


# Load and visualize the PDB files for the MOFs.
def load_PDB_file(molecule):
    """Loads PDB file.
    """

    frame_file_name = DATAPATH / f"{molecule}.pdb"

    with open(frame_file_name, "r") as pdb_frame:
        properties_array = pdb_frame.readlines()[1].split()[1:]
    # print(properties_array)
    [A, B, C, alpha_deg, beta_deg, gamma_deg] = [float(parameter) for parameter in properties_array]
    # print([A, B, C, alpha_deg, beta_deg, gamma_deg])
    frame_properties_dict = {"A": A, "B": B, "C": C, "alpha_deg": alpha_deg, \
        "beta_deg": beta_deg, "gamma_deg": gamma_deg}
    #Properties of the frame. they need to be defined somewhere in the code.
    frame_cell_parameter_matrix = make_cell_parameter_matrix(frame_properties_dict)
    frame_dims_perpendicular = frame_cell_parameter_matrix.diagonal()

    df_frame = pd.read_table(frame_file_name, delim_whitespace=True, skiprows=2, 
        usecols=[1, 2, 4, 5, 6], names=["AtomNum", "AtomType", "x", "y", "z"])

    return df_frame, A, B, C, frame_properties_dict, frame_cell_parameter_matrix


def load_water_movie_file(file_name):
    """Loads the aggregated snapshot PDB of water molecule in the MOF framework.
    """
    
    movie_file_name = DATAPATH / file_name

    O_pos_df = pd.read_table(movie_file_name, \
            delim_whitespace=True, skiprows=1, usecols=[4,5,6], names=["Ow_x", "Ow_y", "Ow_z"]) #Read directly from a bash output. 

    return O_pos_df


def make_3d_histograms(O_pos_df, A, B, C):
    """Creates 3D histograms for the water molecules in the MOF framework.
    """

    O_pos_array = O_pos_df[["Ow_x", "Ow_y", "Ow_z"]].values

    bins = [tuple(np.arange(0, A, 1)), 
    tuple(np.arange(0, B, 1)), tuple(np.arange(0, C, 1))]
    hist, edges = np.histogramdd(O_pos_array, bins=bins, density=True)
    hist[hist==0] = np.exp(-100)
    #compute the mid-points. 
    mid_points = [(edge_val[:-1] + edge_val[1:]) /2 for edge_val in edges]

    #create 3D mesh to match with the 3D histograms (hopefully it matches)
    #mid_point_mesh = np.meshgrid(mid_points[0], mid_points[1], mid_points[2])
    mid_point_mesh = np.meshgrid(mid_points[0], mid_points[1], mid_points[2], indexing="ij")
    return hist, mid_point_mesh


def create_low_energy_mask(mid_point_mesh, hist):
    #Create an array of low free energies and compute the graph. 
    mesh_array = np.column_stack((mid_point_mesh[0].flatten(), mid_point_mesh[1].flatten(),\
        mid_point_mesh[2].flatten(), -np.log(hist.flatten()) ))
    mesh_array_df = pd.DataFrame(mesh_array, columns=["x", "y", "z", "FreeEnergy"])
    energy_cutoff=10
    low_energy_mask = mesh_array_df["FreeEnergy"] < energy_cutoff
    mesh_array_df = mesh_array_df[low_energy_mask]

    return mesh_array_df


def visualize_RDF(mid_point_mesh, hist, frame_properties_dict):
    """Compute and visualize the RDF.
    """

    mesh_array_df = create_low_energy_mask(mid_point_mesh, hist)

    fig_rdf = go.Figure()
    #gr, hist_bins = compute_RDF(copy.deepcopy(O_pos_array[1::50, :]), frame_properties_dict)
    gr, hist_bins = compute_RDF(mesh_array_df[["x", "y", "z"]].values, frame_properties_dict, \
        dr=1, sample_size=min(mesh_array_df.shape[0], 1000))

    #fig_rdf.add_trace(go.Scatter(x=hist_bins[1:], y=gr, mode="markers+lines"))
    #fig_rdf.update_xaxes(title_text="r", range=(2.5, 12))
    #fig_rdf.update_yaxes(title_text="g(r)", range=(0, 5))

    fig_rdf.add_trace(go.Scatter(x=hist_bins[1:], y=gr, mode="markers+lines"))
    fig_rdf.update_xaxes(title_text="r", range=(0, 12))
    fig_rdf.update_yaxes(title_text="g(r)", range=(-0.5, 5))
    fig_rdf.update_layout(autosize=False) #width=7 * 96 * 0.5, height=6 * 96 * 0.5

    return fig_rdf


# Compute NNN graphs.
def visualize_NNN_graph(mid_point_mesh, hist, frame_cell_parameter_matrix, fig, molecule):

    mesh_array_df = create_low_energy_mask(mid_point_mesh, hist)

    # Here, we will implement the nearest neighbor network. Later, this function 
    # can be transferred to a different file for network utils. 
    sample_size=min(mesh_array_df.shape[0], 1000)
    NNN = make_NNN(mesh_array_df, frame_cell_parameter_matrix, size=sample_size)
    nodes , edges, node_info = NNN

    #Add the network traces
    plot_network_traces(NNN, fig)

    # print(f"Edges: {len(edges)}, Vertices: {node_info.shape[0]}, Beta index: {len(edges) /node_info.shape[0] : .2f}")
    # with open("beta_index_vals.txt", "a+") as outfile:
    #     outfile.write(f"{molecule} Edges: {len(edges)}, Vertices: {node_info.shape[0]},\
    #     Beta index: {len(edges) /node_info.shape[0] : .2f}\n")

    return fig
# Visualize these. 