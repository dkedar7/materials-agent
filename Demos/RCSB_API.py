"""
Creates functions to interact with RCSB API.
"""

import requests
import json
from src import BASE_DIR


def get_RCSB_id(common_name: str):
    """
    Makes the API call to RCSB API.
    """

    query_dict = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": common_name
            }
        },
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 1
            }
        },
        "return_type": "entry"
    }

    query_str = json.dumps(query_dict)

    # Construct the API query URL
    api_url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={query_str}"

    # Make the API request
    response = requests.get(api_url)
    if response.status_code != 200:
        print("Error: Unable to fetch data from RCSB API.")
        return None

    # Parse the response
    data = response.json()
    if 'result_set' in data.keys():
        if len(data['result_set']) > 0 and data['result_set'][0]['score'] > 0.9:
            pdb_id = data['result_set'][0]['identifier']
            return pdb_id
        else:
            print("Error: Molecule not found.")
            return None
    else:
        print("Error: Invalid response from RCSB API.")
        return None


def download_atomic_coordinate_data_of_molecule(RCSB_database_identifier: str):
    """
    Given RCSB database identifier, downloads  /gets its atomic coordinates; in other words, PDB data.
    Uses the RCSB database to first get the database key and then downloads this to the folder.
    Commonly, the RCSB database identifier can be obtained from the common name using the 
    "get_RCSB_identifier" tool.
    """

    # Construct the URL to download the PDB file
    url = f"https://files.rcsb.org/download/{RCSB_database_identifier}.pdb"

    # Make the request to download the PDB file
    response = requests.get(url)
    
    DATAPATH = BASE_DIR / 'Demos/data'
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(f"{DATAPATH}/{RCSB_database_identifier}.pdb", "wb") as file:
            file.write(response.content)
        return True
    else:
        return False
