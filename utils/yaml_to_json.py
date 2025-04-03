import yaml
import json

# Đọc file YAML
with open("input.yaml", "r") as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

# Chuyển sang JSON
json_data = json.dumps(yaml_data, indent=2)

# Ghi ra file JSON
with open("output.json", "w") as json_file:
    json_file.write(json_data)