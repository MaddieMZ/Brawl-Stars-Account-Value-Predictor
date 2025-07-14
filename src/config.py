import numpy as np

FEATURE_WEIGHTS= {
    'limited_skins': 5.0,
    'legendary_skins': 2.0,
    'epic_skins': 1.2,
    'super_rare_skins': 0.8,
    'hypercharges': 2.5,
    'legendary_brawlers': 2.3,
    'mythic_brawlers': 1.7,
    'epic_brawlers': 1.4,
    'star_powers':1.3,
    'gadgets': 1.2,
    'trophy count': 0.7,
    'coins': 0.5,
    'gems': 1.8,
    'bling': 0.3,
    'max_rank':0.7,
    'limited_pins':1.2,
    'limited_sprays':1.2,
    'brawl_pass_titles': 1.4,
    'fame_level':0.3
}

LOG_FEATURES = ['trophy count', 'coins']

CATEGORICAL_FEATURES = {
    'max_rank': ['Bronze','Silver','Gold','Diamond','Mythic','Legendary','Masters','Pro'],
    'fame_level': ['Global','Lunar','Martian','Saturnian','Solar','Meteoric','Alien']
}

DATA_PATHS = {
    'raw_data': 'data/raw_data.csv',
    'processed_data': 'data/processed_data.pkl',
    'model': 'models/brawl_account_worth_model.pkl',
    'encoder': 'models/encoder.pkl',
    'scaler': 'models/scaler.pkl'
}

