import pandas as pd
from src.model_training import train_model
from src.prediction import ValuePredictor

def collect_sample_data():
    # Sample data collection logic
    sample_data = {
    'limited_skins': 12,
    'legendary_skins': 6,
    'epic_skins': 14,
    'super_rare_skins': 18,
    'hypercharges': 53,
    'legendary_brawlers': 7,
    'mythic_brawlers': 10,
    'epic_brawlers': 6,
    'star_powers':91,
    'gadgets': 96,
    'trophy count': 70000,
    'coins': 1545,
    'gems': 160,
    'bling': 3100,
    'max_rank':'Diamond',
    'limited_pins':14,
    'limited_sprays':23,
    'brawl_pass_titles': 4,
    'fame_level':'Saturnian',
    'account_worth': 0
    }
    return sample_data

def main():
    print("Starting model training...")
    train_model()

    predictor= ValuePredictor()

    sample_data = collect_sample_data()
    predicted_value=predictor.predict(sample_data)

    print(f"Predicted account value: {predicted_value:.2f}")



if __name__ == "__main__":
    main()
