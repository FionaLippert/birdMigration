import yaml

with open('config.yml') as f:
    config = yaml.load(f)

print([f.path for f in os.scandir(config['datadir']) if f.is_dir()])
