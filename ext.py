import json

# Function to extract link tuples from the provided JSON file
def extract_link_tuples_from_json(json_file_path):
    # Open and read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    print(data)
    # Extracting all tuples from p1, p2, and p3 of all paths
    link_tuples = []
    for path, details in data["paths"].items():
        link_tuples.extend(details["p1"])
        link_tuples.extend(details["p2"])
        link_tuples.extend(details["p3"])
    
    return link_tuples

# Specify the path to your JSON file
json_file_path = '16_CRs_paths.json'

# Extract the tuples
link_tuples = extract_link_tuples_from_json(json_file_path)

# Print each tuple in the desired format
for link in link_tuples:
    print(link.strip("()").replace(",", ""))
