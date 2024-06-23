# Define the tools here.
# Also includes all the utility functions built using AdsVis, RDKkit, GROMACS, etc.

from langchain.agents import tool
import matplotlib.pyplot as plt
from Demos.create_visualization import render_molecule
from Demos.create_adsorption_visualization import load_PDB_file, load_water_movie_file,\
    make_3d_histograms, visualize_RDF, visualize_NNN_graph
from Demos.plotly_utils_test import plot_molecule, plot_water_movie

from langchain_openai import ChatOpenAI
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors


from Demos.RCSB_API import get_RCSB_id, download_atomic_coordinate_data_of_molecule
import dash_bio
from dash_bio.utils import pdb_parser as parser, mol3dviewer_styles_creator as sparser
from src import BASE_DIR

from embedchain import App

@tool
def create_3D_visualization_using_plotly(molecule: str):
    """Returns a 3D visualization of the molecule using Plotly.
    This creates an interactive figure.
    """

    try:
        # Loads PDB file
        df_frame, A, B, C, frame_properties_dict, frame_cell_parameter_matrix =\
            load_PDB_file(molecule)

        # Plotting the frame molecule. 
        fig = plot_molecule(molecule, df_frame)
        fig.update_layout(template="simple_white")

        return fig

    except Exception:
        # Try Dash's functionality to load the PDB file.
        try:
        #if True:

            data_path = BASE_DIR / 'Demos/data' / f"{molecule}.pdb"
            data_path = str(data_path)

            mol_style = "stick" #'cartoon'
            color_style = 'residue'
            # Create the model data from the decoded contents
            pdb = parser.PdbParser(data_path)
            mdata = pdb.mol3d_data()

            # Create the cartoon style from the decoded contents
            data_style = sparser.create_mol3d_style(
                mdata.get("atoms"),
                mol_style,
                color_style
            )

            molecule_viewer = dash_bio.Molecule3dViewer(
                    id='mol-3d',
                    selectionType='atom',
                    modelData=mdata,
                    styles=data_style,
                    selectedAtomIds=[],
                    backgroundOpacity='0',
                    atomLabelsShown=False,
                )

            return molecule_viewer
        except:
            return "The tool call failed."
            

@tool
def create_3D_heatmap_of_locations(framework_molecule, file_name):
    """Returns a 3D heatmap visualization of the locations specified in the txt file.
    """

    try:
        molecule = framework_molecule

        # Loads PDB file
        df_frame, A, B, C, frame_properties_dict, frame_cell_parameter_matrix =\
            load_PDB_file(molecule)

        # Plotting the frame molecule. 
        fig = plot_molecule(molecule, df_frame)

        O_pos_df = load_water_movie_file(file_name)

        # Compute the 3D histogram.
        hist , mid_point_mesh = make_3d_histograms(O_pos_df, A, B, C)

        # Visualize the movie file inside the frame.
        plot_water_movie(fig, mid_point_mesh, hist)

        return fig
    
    except Exception:
        return "The tool call failed."


@tool
def create_and_visualize_RDF_of_coordinate_data(framework_molecule, file_name):
    """Creates and visualized RDF or Radial Distribution Function of the coordinates
    mentioned in the file. In order to accommodate triclinic periodic boundary conditions,
    the framrwork molecule in which adsorption occurs is also specified.
    Input argument: Framework molecule, file name of the coordinates.
    """

    try:
        molecule = framework_molecule

        # Loads PDB file
        df_frame, A, B, C, frame_properties_dict, frame_cell_parameter_matrix =\
            load_PDB_file(molecule)

        O_pos_df = load_water_movie_file(file_name)

        # Compute the 3D histogram.
        hist , mid_point_mesh = make_3d_histograms(O_pos_df, A, B, C)

        # Compute the RDF.
        fig_rdf = visualize_RDF(mid_point_mesh, hist, frame_properties_dict)
        fig_rdf.update_layout(template="simple_white")

        return fig_rdf
    
    except Exception:
        return "The tool call failed."


@tool
def create_and_visualize_network_of_preferred_adsoprtion_sites(framework_molecule, file_name):
    """Creates and visualizes network of preferred adsorption sites.
    This is a nearest neighbor network.
    """

    try:
        molecule = framework_molecule

        # Loads PDB file
        df_frame, A, B, C, frame_properties_dict, frame_cell_parameter_matrix =\
            load_PDB_file(molecule)

        # Plotting the frame molecule. 
        fig = plot_molecule(molecule, df_frame)

        O_pos_df = load_water_movie_file(file_name)

        # Compute the 3D histogram.
        hist , mid_point_mesh = make_3d_histograms(O_pos_df, A, B, C)

        # Compute the network and show it.
        fig_with_network = visualize_NNN_graph(mid_point_mesh, hist,
                                                frame_cell_parameter_matrix, fig, molecule)

        return fig_with_network
    
    except Exception:
        return "The tool call failed."


@tool
def return_regular_reponse(prompt: str) -> str:
    """
    If none of the previous tools are applicable to the query at hand,
    supplies the prompt to ChatGPT and returns the reponse as is.
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    messages = [
        ("system", "You are a helpful AI assistant that helps with answering questions."),
        ("human", prompt),  
    ]
    response = llm.invoke(messages)

    return response


@tool
def create_2D_visualization_from_SMILES(SMILES_string):
    """
    Get a visual 2D representation of a molecule from its SMILES string.
    """

    m = Chem.MolFromSmiles(SMILES_string)

    img = Draw.MolToImage(m)

    return img


@tool
def get_similarity_of_molecules(SMILES_str_1, SMILES_str_2):
    """
    Get similarity between two SMILES strings using the RDKit fingerprints and Tanimoto similarity.
    """

    ms = [Chem.MolFromSmiles(SMILES_str_1), Chem.MolFromSmiles(SMILES_str_2)]
    fpgen = rdFingerprintGenerator.GetRDKitFPGenerator()
    fps = [fpgen.GetFingerprint(x) for x in ms]
    similarity = DataStructs.TanimotoSimilarity(fps[0], fps[1])

    return {
        "Status" : "Computed successfully",
        "Similarity" : similarity
    }


@tool
def download_PDB_file_for_molecule(common_name: str):
    """
    Downloads the PDB file for a molecule given its common name from RCSB database.
    The function first generates the RCSB database identifier using the common name.
    Then, it is properly generated, it downloads the file.
    It will return whether the download was successful or not as well as the 
    RCSB id of the molecule.
    """

    RCSB_database_identifier = get_RCSB_id(common_name)

    if RCSB_database_identifier:
        file_download_success = download_atomic_coordinate_data_of_molecule(RCSB_database_identifier)
        if file_download_success:
            return {
                "Status" : "Download successful",
                "RCSB_database_identifier" : RCSB_database_identifier
            }
        else:
            return {
                "status" : "File download failed",
                "RCSB_database_identifier" : RCSB_database_identifier
            }
    else:
        return "The tool call failed. Unable to retrieve RCSB database identifier."


@tool
def get_description_of_molecule_from_SMILES(SMILES_string):
    """
    Get a description of a molecule from its SMILES string.
    Compute molecular descriptors using RDKit.
    """

    m = Chem.MolFromSmiles(SMILES_string)

    desc = Descriptors.CalcMolDescriptors(m)


    short_listed_props = ['qed', 'MolWt', 'ExactMolWt', 'NumValenceElectrons',
    'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
    'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
    'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
    'RingCount', 'MolLogP']
    
    
    desc = {key: val for key, val in desc.items() if key in short_listed_props}

    return desc


@tool
def find_answer_from_urls(urls: list, question: str) -> str:
    "Given webpage URLs and a question, find the answer from the webpages."

    app = App()

    # Embed online resources
    _ = [app.add(u) for u in urls]

    # Query the app
    answer = app.query(question)
    return answer


tools = [
    create_3D_visualization_using_plotly,
    create_3D_heatmap_of_locations,
    create_and_visualize_RDF_of_coordinate_data,
    create_and_visualize_network_of_preferred_adsoprtion_sites,
    return_regular_reponse,
    create_2D_visualization_from_SMILES,
    get_similarity_of_molecules,
    download_PDB_file_for_molecule,
    get_description_of_molecule_from_SMILES,
    find_answer_from_urls
]