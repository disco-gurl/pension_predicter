import importlib
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# Load client data
filename = "client_data"
client_data = importlib.import_module(filename)
agent_dict = client_data.agent_dict

# Define age offsets for future predictions
age_offsets = [1, 5, 10]

# Financial and economic variables
bank_interest_rate = 2.0
competitor_interest_rate = 1.8
inflation_rate = 2.12
employment_rate = 80

decision = [True, False]

baseInflation = 2.12 
counter = 0

# User input
age = int(input("Please enter your age: "))

# Simulating inflation changes
inflation = baseInflation
for i in range(age):
    if age < 66:
        randomNum = random.randrange(1, 4)
        randomDec = random.choice(decision)
        inflation += randomNum if randomDec else -randomNum

# Function to determine pension and savings rate based on age
def get_pension_savings_rate(age):
    if 16 <= age <= 17:
        return 0.10, 0.10
    elif 18 <= age <= 21:
        return 0.10, 0.14
    elif 22 <= age <= 29:
        return 0.15, 0.20
    elif 30 <= age <= 39:
        return 0.20, 0.24
    elif 40 <= age <= 49:
        return 0.25, 0.28
    elif 50 <= age <= 59:
        return 0.30, 0.36
    elif 60 <= age <= 105:
        return 0.35, 0.44
    return 0, 0  # Default in case of invalid age

# Function to generate simulated data
def simulations(lower, upper):
    simulated_data = {}
    number_of_simulations = 100
    filter_data = {key: value for key, value in agent_dict.items() if (lower <= value["age"] <= upper)}

    # Extract key financial metrics
    incomes = [data['income_after_tax'] for data in filter_data.values()]
    pensions = [data['pension_account'] for data in filter_data.values()]
    savings = [data['savings_account'] for data in filter_data.values()]
    years_bank = [data['years_with_bank'] for data in filter_data.values()]
    current = [data['current_account'] for data in filter_data.values()]

    # Compute means and standard deviations
    mean_income, std_income = np.mean(incomes), np.std(incomes)
    mean_pension, std_pension = np.mean(pensions), np.std(pensions)
    mean_savings, std_savings = np.mean(savings), np.std(savings)
    mean_year_bank, std_year_bank = np.mean(years_bank), np.std(years_bank)
    mean_current, std_current = np.mean(current), np.std(current)

    # Generate simulated data
    for i in range(number_of_simulations):
        simulated_data[f"Simulated Person {i+1}"] = {
            "income_after_tax": max(0, round(np.random.normal(mean_income, std_income))),
            "pension_account": max(0, round(np.random.normal(mean_pension, std_pension))),
            "savings_account": max(0, round(np.random.normal(mean_savings, std_savings))),
            "current_account": max(0, round(np.random.normal(mean_current, std_current))),
            "years_with_bank": max(0, round(np.random.normal(mean_year_bank, std_year_bank)))
        }
        
    return simulated_data

# Model training function
def probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate):
    filtered = {key: value for key, value in agent_dict.items() if (age_group_lower <= value["age"] <= age_group_upper)}
    data = pd.DataFrame.from_dict(filtered, orient="index")

    data["Bank Interest"] = bank_interest_rate
    data["Competitors Interest"] = competitor_interest_rate
    data["Inflation"] = inflation_rate
    data["Employment"] = employment_rate

    x_factors = data[["income_after_tax", "savings_account", "current_account", "pension_account", "years_with_bank", "Inflation", "Employment", "Bank Interest", "Competitors Interest"]]
    y_factors = data["left_bank"]

    scaler = MinMaxScaler()
    x_scale = scaler.fit_transform(x_factors)

    x_train, x_test, y_train, y_test = train_test_split(x_scale, y_factors, test_size=0.1, random_state=42)

    model = tf.keras.models.Sequential([
        layers.InputLayer(input_shape=(x_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        l
