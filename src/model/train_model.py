import joblib
import os
from lifelines import CoxPHFitter, KaplanMeierFitter

from ..data import data_processing

def train_model(machine_path='data/machine_data.csv', maintenance_path='data/maintenance_data.csv'):
    """
    Train/Retrain a Cox Proportional Hazards model using the given machine and maintenance data.

    Args:
    - machine_path (str): Path to the machine data.
    - maintenance_path (str): Path to the maintenance data.

    Returns:
    - cph (CoxPHFitter): Trained Cox model.
    - used_categories (dict): Dictionary of categories used during training.
    """

    train_df, used_categories = data_processing.prepare_training_data(machine_path=machine_path, maintenance_path=maintenance_path)

    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='Time Since Last Fix', event_col='Event')
    cph.print_summary()

    return cph, used_categories

def save_model(model, used_categories, model_path='saved_model/cox_model.pkl', categories_path='saved_model/used_categories.pkl'):
    """
    Save the trained Cox Proportional Hazards model and used categories.

    Args:
    - cph (CoxPHFitter): Trained Cox model.
    - used_categories (dict): Dictionary of categories used during training.
    - model_path (str): Path to save the model.
    - categories_path (str): Path to save the used categories.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(categories_path), exist_ok=True)

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save the used categories
    joblib.dump(used_categories, categories_path)
    print(f"Used categories saved to {categories_path}")

def load_model(model_path='saved_model/cox_model.pkl', categories_path='saved_model/used_categories.pkl'):
    """
    Load the trained Cox Proportional Hazards model and used categories.

    Args:
    - model_path (str): Path to load the model from.
    - categories_path (str): Path to load the used categories from.

    Returns:
    - cph (CoxPHFitter): Trained Cox model.
    - used_categories (dict): Dictionary of categories used during training.
    """
    if not os.path.exists(model_path) or not os.path.exists(categories_path):
        raise FileNotFoundError(f"Model or categories file not found. Ensure {model_path} and {categories_path} exist.")

    # Load the model
    cph = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Load the used categories
    used_categories = joblib.load(categories_path)
    print(f"Used categories loaded from {categories_path}")

    return cph, used_categories
