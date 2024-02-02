import yaml
import pickle
with open('.github\workflows\workflow.yaml', 'r') as f:
    data = yaml.safe_load(f)
    pickle.dump(data, open('data.py', 'wb')) 