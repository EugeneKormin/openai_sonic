import configparser


config = configparser.ConfigParser()
config.read('config.ini')

games = int(config['data']['games'])
attemps = int(config['data']['attemps'])
generations = int(config['data']['generations'])
env_name = config['data']['env_name']
min_epochs = int(config['data']['min_epochs'])
max_epochs = int(config['data']['max_epochs'])
max_evals = int(config['data']['max_evals'])
threshold_coeff = float(config['data']['threshold_coeff'])
threshold = float(config['data']['threshold'])

