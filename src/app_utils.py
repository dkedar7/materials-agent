from fast_dash import dbc, dcc, dmc

from dash_iconify import DashIconify
from dash import Patch
from dash.exceptions import PreventUpdate

import time

def Chatify(query_response_dict):
    "Convert a dictionary into a Chat component"

    if not isinstance(query_response_dict, dict):
        raise TypeError("Chat component requires a dictionary output ('query': ..., 'response': ...).")
    
    if query_response_dict["query"] == "":
        raise PreventUpdate

    ## Chat component
    input_component = dbc.Row(
        dmc.Text(
            [query_response_dict["query"]],
            align="end",
            style={
                "padding": "1% 1%",
                "max-width": "80%",
                "backgroundColor": "#E8EBFA",
            },
            className="border rounded shadow-sm m-3 col-auto",
        ),
        align="center",
        justify="end",
    )

    output_component = dbc.Row(
        [
            dmc.Text(
                [
                    dbc.Col(
                        DashIconify(
                            icon="ic:baseline-question-answer",
                            color="#910517",
                            width=30,
                        ),
                        class_name="pb-2",
                    ),
                    dcc.Markdown(query_response_dict["response"]),
                    dcc.Graph(id=str(time.time()), figure=query_response_dict.get("plotly_figure"), 
                              style={"max-height":"200%", "width":"100%"}) if "plotly_figure" in query_response_dict else ""
                ],
                align="start",
                style={
                    "padding": "1% 1%",
                    "max-width": "98%",
                    "backgroundColor": "#F9F9F9",
                },
                className="border rounded shadow-sm m-3",
            )
        ],
        align="start",
        justify="start",
    )

    chat_output = Patch()
    chat_output.prepend(input_component)
    chat_output.prepend(output_component)

    return chat_output


ABOUT_MARKDOWN = """### Motivation
Large Language Models (LLMs) excel at providing guidance and suggesting directions for exploration. However, in the realm of molecular discovery, they occasionally produce inaccurate or fabricated information. Additionally, given the plethora of tools available for molecular discovery, selecting the appropriate tool for a specific task can be daunting. This involves surveying potential tools, understanding their documentation, evaluating their suitability, and implementing them. Our goal is to streamline this process with Materials Agent: an open-source LLM agent equipped with both standard and custom-designed cheminformatics tools.

### What is Materials Agent?
Materials Agent is an LLM agent enhanced with a selection of open-source tools and bespoke utilities designed to facilitate common molecular discovery workflows.

Specifically, the prototype should enable the following functionalities:

- Provide 2D and 3D visualizations of molecules.
- Offer computation and visualization of 3D heatmaps to represent results from adsorption studies, with potential extensions to other applications.
- Assist in the generation of input files for widely utilized simulation techniques, including DFT (Quantum Espresso), Molecular Dynamics (GROMACS), and Monte Carlo (RASPA) simulations.
- Enable visualization and detailed descriptions of molecules using common tools available in RDKit.

For questions and suggestions, please contact [Archit Datar](mailto:architdatar.com) or [Kedar Dabhadkar](https://linkedin.com/in/dkedar7)."""