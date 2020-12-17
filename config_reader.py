import configparser


config = configparser.ConfigParser()
config.read('config.ini')

LR = float(config['data']['LR'])
games = int(config['data']['games'])
attemps = int(config['data']['attemps'])
generations = int(config['data']['generations'])
env_name = config['data']['env_name']
elitism_factor = float(config['data']['elitism_factor'])
