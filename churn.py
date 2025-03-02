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
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    
    return model, scaler

# Predict customer likelihood of leaving the bank
def predictability(model, scaler, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate, customer):
    analysis_customer = pd.DataFrame({
        "income_after_tax": [customer["income_after_tax"]],
        "savings_account": [customer["savings_account"]],
        "current_account": [customer["current_account"]],
        "pension_account": [customer["pension_account"]],
        "years_with_bank": [customer["years_with_bank"]],
        "Inflation": [inflation_rate],
        "Employment": [employment_rate],
        "Bank Interest": [bank_interest_rate],
        "Competitors Interest": [competitor_interest_rate] 
    })

    scaled_customer = scaler.transform(analysis_customer)
    probability = model.predict(scaled_customer)
    
    return probability

# Select the age group and run simulations
age_groups = [(16, 17, 85.95), (18, 21, 85.95), (22, 29, 96.05), 
              (30, 39, 97.42), (40, 49, 97.42), (50, 59, 97.47), (60, 90, 97.47)]

for lower, upper, emp_rate in age_groups:
    if lower <= age <= upper:
        employment_rate = emp_rate
        simulation_data = simulations(lower, upper)
        model, scaler = probabality(lower, upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
        break

# Run simulations for a selected random person
random_person = random.choice(list(simulation_data.values()))
cumulative_savings = random_person['savings_account']
cumulative_pension = random_person['pension_account']
final_age = age  # Track the last calculated age

for offset in age_offsets:
    final_age = age + offset
    if final_age > 105:
        break

    updated_income = random_person['income_after_tax']
    pension_rate, savings_rate = (0.35, 0.44) if final_age >= 60 else (0.30, 0.36) 

    cumulative_pension += updated_income * pension_rate
    cumulative_savings += updated_income * savings_rate

    print(f"\nA customer of age {final_age}:")
    print(f"Their pension will be £{cumulative_pension:.2f}.")
    print(f"Their savings will be £{cumulative_savings:.2f}.")

# If last calculated age is still under 66, calculate for 66 years old too
if final_age < 66:
    final_age = 66
    pension_rate, savings_rate = 0.35, 0.44  # Use rates for 60+
    cumulative_pension += updated_income * pension_rate
    cumulative_savings += updated_income * savings_rate

    print(f"\nA customer of age {final_age}:")
    print(f"Their pension will be £{cumulative_pension:.2f}.")
    print(f"Their savings will be £{cumulative_savings:.2f}.")


