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
agent_dict = client_data.agent_dict

age_offsets = [1, 5, 10]  # Add +1, +4, +5 to the original age


age_groups = [(16, 17, 85.95), (18, 21, 85.95), (22, 29, 96.05), 
              (30, 39, 97.42), (40, 49, 97.42), (50, 59, 97.47), (60, 90, 97.47)]

simulated_person = {}

bank_interest_rate = 2.0
competitor_interest_rate = 1.8
inflation_rate = 2.12
   
employment_rate = 80

decision = [True, False]

baseInflation = 2.12 
counter = 0

age = int(input("Please enter your age: "))

inflation = baseInflation
for i in range(age):
    if age < 66:
        randomNum = random.randrange(1, 4)
        randomDec = random.choice(decision)
        inflation += randomNum if randomDec else -randomNum

def get_pension_savings_rate(person_age):
    if 16 <= person_age <= 17:
        return 0, 10
    elif 18 <= person_age <= 21:
        return 10, 14
    elif 22 <= person_age <= 29:
        return 15, 20
    elif 30 <= person_age <= 39:
        return 20, 24
    elif 40 <= person_age <= 49:
        return 25, 28
    elif 50 <= person_age <= 59:
        return 30, 36
    else:
        return 35, 44 

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
    
    return float(probabilty[0])



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

random_person = random.choice(list(simulation_data.values()))
cumulative_savings = random_person['savings_account']
cumulative_pension = random_person['pension_account']
cumulative_current_account = random_person['current_account']
years_with_bank = random_person['years_with_bank']

final_age = age  # Track the last calculated age

for offset in age_offsets:
    final_age += offset
    if final_age > 105:
        break

    pension_rate, savings_rate = get_pension_savings_rate(final_age)

    pension_contribution = random_person['income_after_tax'] * pension_rate
    savings_contribution = random_person['income_after_tax'] * savings_rate

    cumulative_pension += pension_contribution
    cumulative_savings += savings_contribution
    cumulative_current_account += random_person['income_after_tax'] - (pension_contribution + savings_contribution)

    years_with_bank += offset

    print(f"\nThe customer of age {final_age}:")
    print(f"Their pension will be £{cumulative_pension:.2f}.")
    print(f"Their savings will be £{cumulative_savings:.2f}.")
    customer = {'age': final_age, 'income_after_tax': random_person['income_after_tax'], 'years_with_bank': years_with_bank, 'savings_account': cumulative_savings, 'current_account': cumulative_current_account, 'pension_account': cumulative_pension}
    number2 = predictability(model, scaler, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate, customer)
    print("The probablity of them leaving is ", number2 , "%.")
    percentage = round(number2 * 100, 2)
    
    simulated_person.update({'age': final_age, 'income_after_tax': random_person['income_after_tax'], 'savings_account': cumulative_savings, 'pension_account': cumulative_pension, 'likehood_to_leave': percentage})
    

# If last calculated age is still under 66, calculate for 66 years old too
while final_age < 66:
    final_age += 1
    pension_rate, savings_rate = get_pension_savings_rate(final_age)

    pension_contribution = random_person['income_after_tax'] * pension_rate
    savings_contribution = random_person['income_after_tax'] * savings_rate

    cumulative_pension += pension_contribution
    cumulative_savings += savings_contribution
    cumulative_current_account += random_person['income_after_tax'] - (pension_contribution + savings_contribution)

    years_with_bank += 1


if final_age == 66:
    print(f"\nThe customer of age {final_age}:")
    print(f"Their pension will be £{cumulative_pension:.2f}.")
    print(f"Their savings will be £{cumulative_savings:.2f}.")
    customer = {'age': final_age, 'income_after_tax': random_person['income_after_tax'], 'years_with_bank': years_with_bank, 'savings_account': cumulative_savings, 'current_account': cumulative_current_account, 'pension_account': cumulative_pension}
    number2 = predictability(model, scaler, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate, customer)
    print("The probablity of them leaving is ", number2 , "%.")
    simulated_person.update({'age': final_age, 'income_after_tax': random_person['income_after_tax'], 'savings_account': cumulative_savings, 'pension_account': cumulative_pension, 'likehood_to_leave': percentage})
    
    
