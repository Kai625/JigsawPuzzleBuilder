"""
Author:         Mr. H. van der Westhuizen
Date opened:    11 September 2021
Student Number: u18141235
Project number: HG1

Read write Json testing.
"""

import json

dictionary_data = {"a": True, "b": 2}

# a_file = open("data.json", "w")
# json.dump(dictionary_data, a_file, indent=4)
# a_file.close()

json_file = open("data.json")
variables = json.load(json_file)
print(variables)
json_file.close()
