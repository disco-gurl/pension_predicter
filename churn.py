import importlib
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

filename = "client_data"
client_data = importlib.import_module(filename)
clientdict = client_data.agent_dict

agent_dict = {0: {'age': 60, 'income_after_tax': 17861.4, 'years_with_bank': 45, 'savings_account': 257204.16, 'current_account': 11057.58, 'pension_account': 0.0, 'left_bank': 1}, 1: {'age': 54, 'income_after_tax': 19942.8, 'years_with_bank': 39, 'savings_account': 233330.76, 'current_account': -12933.96, 'pension_account': 940324.48, 'left_bank': 0}, 2: {'age': 72, 'income_after_tax': 17543.4, 'years_with_bank': 57, 'savings_account': 319991.62, 'current_account': 146665.55, 'pension_account': 0.0, 'left_bank': 0}, 3: {'age': 31, 'income_after_tax': 15480.0, 'years_with_bank': 16, 'savings_account': 59443.2, 'current_account': 31535.1, 'pension_account': 88713.2, 'left_bank': 0}, 4: {'age': 23, 'income_after_tax': 15772.8, 'years_with_bank': 8, 'savings_account': 25236.48, 'current_account': 34517.48, 'pension_account': 14991.22, 'left_bank': 0}, 5: {'age': 52, 'income_after_tax': 18956.4, 'years_with_bank': 37, 'savings_account': 210416.04, 'current_account': 83765.17, 'pension_account': 735642.04, 'left_bank': 0}}

age_offsets = [0, 1, 4, 5]  # Add +1, +4, +5 to the original age

bank_interest_rate = 2.0
competitor_interest_rate = 1.8
inflation_rate = 2.12
   
employment_rate = 80

decision = [True, False]

baseInflation = 2.12 
counter = 0

age = int(input("Please enter your age"))

inflation = baseInflation
for i in range(age):
    if age < 66:
        randomNum = random.randrange(1, 4)
        randomDec = random.choice(decision)
        inflation += randomNum if randomDec else -randomNum



def simulations(lower, upper):
    simulated_data = {}
    number_of_simulations = 100
    filter_data = {
    key: value for key, value in agent_dict.items() 
    if (lower <= value["age"] <= upper)
    }
    
    incomes = [data['income_after_tax'] for data in filter_data.values()]
    pensions = [data['pension_account'] for data in filter_data.values()]
    savings = [data['savings_account'] for data in filter_data.values()]
    years_bank = [data['years_with_bank'] for data in filter_data.values()]
    current = [data['current_account'] for data in filter_data.values()]
    
    mean_income = np.mean(incomes)
    std_income = np.std(incomes)
    mean_pension = np.mean(pensions)
    std_pension = np.std(pensions)
    mean_savings = np.mean(savings)
    std_savings = np.std(savings)
    mean_year_bank = np.mean(years_bank)
    std_year_bank = np.std(years_bank)
    mean_current = np.mean(current)
    std_current = np.std(current)
    
    for i in range(number_of_simulations):
        simulated_income = max(0, round(np.random.normal(loc=mean_income, scale=std_income)))
        simulated_pension = max(0, round(np.random.normal(loc=mean_pension, scale=std_pension)))
        simulated_savings = max(0, round(np.random.normal(loc=mean_savings, scale=std_savings)))
        simulated_year_bank = max(0, round(np.random.normal(loc=mean_year_bank, scale=std_year_bank)))
        simulated_current = max(0, round(np.random.normal(loc=mean_current, scale=std_current)))
        
        simulated_data[f"Simulated Person {i+1}"] = {
            "income_after_tax": simulated_income,
            "pension_account": simulated_pension,
            "savings_account" : simulated_savings,
            "current_account" : simulated_current,
            "years_with_bank" : simulated_year_bank 
            
        }
        
    return simulated_data

def probabality (age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate):


    filtered = {
        key: value for key, value in agent_dict.items() 
        if (age_group_lower <= value["age"] <= age_group_upper)
    }

    data = pd.DataFrame.from_dict(filtered, orient="index")

    data["Bank Interest"] = bank_interest_rate
    data["Competitors Interest"] = competitor_interest_rate
    data["Inflation"] = inflation_rate
    data["Employment"] = employment_rate


    x_factors =  data.loc[:,["income_after_tax", "savings_account", "current_account" , "pension_account" ,"years_with_bank", "Inflation", "Employment" ,"Bank Interest", "Competitors Interest"]]
    y_factors = data["left_bank"]

    scaler = MinMaxScaler()
    x_scale = scaler.fit_transform(x_factors)


    x_train, x_test, y_train, y_test = train_test_split(x_scale, y_factors, test_size=0.1, random_state= 42)

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


    model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size = 32)
    
    return model, scaler

    # Predict on your testing set

def predictability (model, scaler, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate, customer):
  
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
    
    analysis_customer = analysis_customer[["income_after_tax", "savings_account", "current_account", "pension_account", "years_with_bank", "Inflation", "Employment", "Bank Interest", "Competitors Interest"]]

   
    scaled_customer = scaler.transform(analysis_customer)


    probabilty = model.predict(scaled_customer)
    
    return probabilty



if 16 <= age <= 17:
    age_group_lower = 16
    age_group_upper = 17
    employment_rate = 85.95
    simulation_data = simulations(age_group_lower, age_group_upper)
    model, scaler = probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
elif 18 <= age <= 21:
    age_group_lower = 18
    age_group_upper = 21
    employment_rate = 85.95
    simulation_data = simulations(age_group_lower, age_group_upper)
    model, scaler = probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
elif 22 <= age <= 29:
    age_group_lower = 22
    age_group_upper = 29
    employment_rate = 96.05
    simulation_data = simulations(age_group_lower, age_group_upper)
    model, scaler = probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
elif 30 <= age <= 39:
    age_group_lower = 30
    age_group_upper = 39
    employment_rate = 97.42
    simulation_data = simulations(age_group_lower, age_group_upper)
    model, scaler = probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
elif 40 <= age <= 49:
    age_group_lower = 40
    age_group_upper = 49
    employment_rate = 97.42
    simulation_data = simulations(age_group_lower, age_group_upper)
    model, scaler = probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
elif 50 <= age <= 59:
    age_group_lower = 50
    age_group_upper = 59
    employment_rate = 97.47
    simulation_data = simulations(age_group_lower, age_group_upper)
    model, scaler = probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
else:
    age_group_lower = 60
    age_group_upper = 90
    employment_rate = 97.47
    simulation_data = simulations(age_group_lower, age_group_upper)
    model, scaler = probabality(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
    
for customer in simulation_data.values():
    number = predictability(model, scaler, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate, customer)
    if number > 0.5:
        counter += 1


print("The likelihood of people aged " , age , " leaving the bank" , counter, "%")


if age < 66:
    random_person = random.choice(list(simulation_data.values()))
    
    for offset in age_offsets:
        current_age = age + offset  # Adjust age with the offset
        if current_age > 105:  
            break  # Prevent unnecessary calculations for ages beyond 105

        # Retrieve simulation data for the current age
        if current_age > 105:  
            break 

        updated_income = random_person['income_after_tax']
        updated_years_in_bank = random_person['years_with_bank']
        updated_current_account = random_person['current_account']

        # Determine pension and savings rates based on age range
        if 16 <= current_age <= 17:
            updated_pension = updated_income * 0.10
            updated_savings = updated_income * 0.10 

        elif 18 <= current_age <= 21:
            updated_pension = updated_income * 0.10
            updated_savings = updated_income * 0.14
        elif 22 <= current_age <= 29:
            updated_pension = updated_income * 0.15
            updated_savings = updated_income * 0.20

        elif 30 <= current_age <= 39:
            updated_pension = updated_income * 0.20
            updated_savings = updated_income * 0.24

        elif 40 <= current_age <= 49:
            updated_pension = updated_income * 0.25
            updated_savings = updated_income * 0.28

        elif 50 <= current_age <= 59:
            updated_pension = updated_income * 0.30
            updated_savings = updated_income * 0.36

        elif 60 <= current_age <= 105:
            updated_pension = updated_income * 0.35
            updated_savings = updated_income * 0.44
        
        # Print the results for each age
        customer = {'age': current_age, 'income_after_tax': updated_income, 'years_with_bank': updated_years_in_bank, 'savings_account': updated_savings, 'current_account': updated_current_account, 'pension_account': updated_pension}
        number2 = predictability(model, scaler, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate, customer)
        print(f"\nA customer of age {current_age}:")
        print(" has the probability of ", number2 , "%")
        print(f". Their pension will be {updated_pension}")
        print(f" and their savings will be {updated_savings}")


