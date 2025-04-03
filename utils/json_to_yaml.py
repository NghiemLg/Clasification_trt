import yaml
import json

# Đọc file JSON
with open("input.json", "r") as json_file:
    json_data = json.load(json_file)

# Chuyển sang YAML
yaml_data = yaml.dump(json_data, default_flow_style=False)

# Ghi ra file YAML
with open("output.yaml", "w") as yaml_file:
    yaml_file.write(yaml_data)