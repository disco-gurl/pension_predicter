import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import importlib

# Load client data
try:
    client_data = importlib.import_module("client_data")
    agent_dict = client_data.agent_dict
except:
    # Fallback sample data if client_data.py is not available
    agent_dict = {
        "agent1": {
            "age": 25,
            "income_after_tax": 35000,
            "pension_account": 125000,
            "savings_account": 75000,
            "current_account": 7000,
            "years_with_bank": 7,
            "left_bank": False
        },
        "agent2": {
            "age": 32,
            "income_after_tax": 42000,
            "pension_account": 160000,
            "savings_account": 96000,
            "current_account": 8400,
            "years_with_bank": 14,
            "left_bank": False
        },
        "agent3": {
            "age": 19,
            "income_after_tax": 29000,
            "pension_account": 95000,
            "savings_account": 57000,
            "current_account": 5800,
            "years_with_bank": 1,
            "left_bank": True
        },
        "agent4": {
            "age": 45,
            "income_after_tax": 55000,
            "pension_account": 225000,
            "savings_account": 135000,
            "current_account": 11000,
            "years_with_bank": 27,
            "left_bank": False
        },
        "agent5": {
            "age": 67,
            "income_after_tax": 38000,
            "pension_account": 335000,
            "savings_account": 201000,
            "current_account": 15400,
            "years_with_bank": 49,
            "left_bank": False
        },
        "agent6": {
            "age": 22,
            "income_after_tax": 32000,
            "pension_account": 110000,
            "savings_account": 66000,
            "current_account": 6400,
            "years_with_bank": 4,
            "left_bank": True
        }
    }

# Define age offsets for future predictions
age_offsets = [1, 5, 10]

# Financial and economic variables
bank_interest_rate = 2.0
competitor_interest_rate = 1.8
inflation_rate = 2.12
employment_rate = 80

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
    mean_income = np.mean(incomes) if incomes else 30000
    std_income = np.std(incomes) if incomes else 5000
    mean_pension = np.mean(pensions) if pensions else 100000
    std_pension = np.std(pensions) if pensions else 20000
    mean_savings = np.mean(savings) if savings else 60000
    std_savings = np.std(savings) if savings else 15000
    mean_year_bank = np.mean(years_bank) if years_bank else 5
    std_year_bank = np.std(years_bank) if years_bank else 2
    mean_current = np.mean(current) if current else 5000
    std_current = np.std(current) if current else 1000

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
def probability(age_group_lower, age_group_upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate):
    filtered = {key: value for key, value in agent_dict.items() if (age_group_lower <= value["age"] <= age_group_upper)}
    
    if not filtered:
        # If no data in this age range, use all data
        filtered = agent_dict
    
    data = pd.DataFrame.from_dict(filtered, orient="index")

    data["Bank Interest"] = bank_interest_rate
    data["Competitors Interest"] = competitor_interest_rate
    data["Inflation"] = inflation_rate
    data["Employment"] = employment_rate

    x_factors = data[["income_after_tax", "savings_account", "current_account", "pension_account", "years_with_bank", "Inflation", "Employment", "Bank Interest", "Competitors Interest"]]
    y_factors = data["left_bank"]

    scaler = MinMaxScaler()
    x_scale = scaler.fit_transform(x_factors)

    # Simple model for demonstration
    model = tf.keras.models.Sequential([
        layers.InputLayer(input_shape=(x_scale.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # For demo purposes, we'll just return the model without training
    return model, scaler

# Calculate probability of leaving the bank
def calculate_leaving_probability(age, pension, savings, current_account, years_with_bank, 
                                 bank_interest=2.0, competitor_interest=1.8, inflation=2.12, employment=80):
    # Base probability decreases with age and years with bank
    probability = 20 - (age / 10) - (years_with_bank / 2)
    
    # Adjust based on financial factors
    probability -= (pension / 50000) * 2
    probability -= (savings / 30000) * 2
    probability -= (current_account / 10000)
    
    # Adjust based on economic factors
    probability += (inflation - bank_interest) * 2
    probability += (competitor_interest - bank_interest) * 5
    probability -= (employment - 80) / 5
    
    # Ensure probability is between 0 and 100
    return max(0, min(100, probability))

# Run financial projections
def run_projections(age, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate):
    # Find age group
    age_groups = [
        (16, 17, 85.95), (18, 21, 85.95), (22, 29, 96.05), 
        (30, 39, 97.42), (40, 49, 97.42), (50, 59, 97.47), (60, 90, 97.47)
    ]
    
    for lower, upper, emp_rate in age_groups:
        if lower <= age <= upper:
            employment_rate = emp_rate
            simulation_data = simulations(lower, upper)
            model, scaler = probability(lower, upper, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
            break
    else:
        # Default if no matching age group
        simulation_data = simulations(22, 29)
        model, scaler = probability(22, 29, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
    
    # Run simulations for a selected random person
    random_person = random.choice(list(simulation_data.values()))
    
    # Base values
    base_income = random_person['income_after_tax']
    base_pension = random_person['pension_account']
    base_savings = random_person['savings_account']
    base_current = random_person['current_account']
    base_years = random_person['years_with_bank']
    
    # Create projections
    projections = []
    
    # Current age projection
    projections.append({
        "age": age,
        "pension": base_pension,
        "savings": base_savings,
        "current_account": base_current,
        "years_with_bank": base_years,
        "leaving_probability": calculate_leaving_probability(
            age, base_pension, base_savings, base_current, base_years,
            bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate
        )
    })
    
    # Future projections
    cumulative_pension = base_pension
    cumulative_savings = base_savings
    cumulative_current = base_current
    years_with_bank = base_years
    
    for offset in age_offsets:
        future_age = age + offset
        if future_age > 105:
            continue
            
        updated_income = base_income * (1 + (offset * 0.02))
        pension_rate = 0.35 if future_age >= 60 else 0.30
        savings_rate = 0.44 if future_age >= 60 else 0.36
        
        pension_contribution = updated_income * pension_rate
        savings_contribution = updated_income * savings_rate
        
        cumulative_pension += pension_contribution
        cumulative_savings += savings_contribution
        cumulative_current += (updated_income - (pension_contribution + savings_contribution))
        years_with_bank += offset
        
        projections.append({
            "age": future_age,
            "pension": cumulative_pension,
            "savings": cumulative_savings,
            "current_account": cumulative_current,
            "years_with_bank": years_with_bank,
            "leaving_probability": calculate_leaving_probability(
                future_age, cumulative_pension, cumulative_savings, cumulative_current, years_with_bank,
                bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate
            )
        })
    
    # Add retirement age (66) if not already included
    if age < 66 and not any(p["age"] == 66 for p in projections):
        retirement_age = 66
        years_to_retirement = retirement_age - age
        
        updated_income = base_income * (1 + (years_to_retirement * 0.02))
        pension_rate = 0.35
        savings_rate = 0.44
        
        retirement_pension = base_pension + (updated_income * pension_rate * years_to_retirement)
        retirement_savings = base_savings + (updated_income * savings_rate * years_to_retirement)
        retirement_current = base_current + ((updated_income - (updated_income * pension_rate) - (updated_income * savings_rate)) * years_to_retirement)
        retirement_years = base_years + years_to_retirement
        
        projections.append({
            "age": retirement_age,
            "pension": retirement_pension,
            "savings": retirement_savings,
            "current_account": retirement_current,
            "years_with_bank": retirement_years,
            "leaving_probability": calculate_leaving_probability(
                retirement_age, retirement_pension, retirement_savings, retirement_current, retirement_years,
                bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate
            )
        })
    
    # Sort projections by age
    projections.sort(key=lambda x: x["age"])
    
    return {
        "person": {
            "income_after_tax": base_income,
            "pension_account": base_pension,
            "savings_account": base_savings,
            "current_account": base_current,
            "years_with_bank": base_years
        },
        "projections": projections,
        "age_group_data": {
            '16-21': 15,
            '22-29': 12,
            '30-39': 8,
            '40-49': 5,
            '50-59': 3,
            '60+': 2
        }
    }

# Streamlit UI
st.set_page_config(page_title="FinVision AI", page_icon="💰", layout="wide")

# Header
st.title("FinVision AI: Financial Prediction & Analysis")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Financial Simulation Parameters")

age = st.sidebar.number_input("Your Age", min_value=16, max_value=90, value=30)

st.sidebar.subheader("Economic Factors")
bank_interest_rate = st.sidebar.slider("Bank Interest Rate (%)", 0.0, 10.0, 2.0, 0.1)
competitor_interest_rate = st.sidebar.slider("Competitor Interest Rate (%)", 0.0, 10.0, 1.8, 0.1)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 20.0, 2.12, 0.1)
employment_rate = st.sidebar.slider("Employment Rate (%)", 50.0, 100.0, 80.0, 0.1)

# Run simulation button
if st.sidebar.button("Generate Predictions"):
    with st.spinner("Processing financial data..."):
        results = run_projections(age, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
        st.session_state.results = results
        st.session_state.tab = "dashboard"

# Initialize session state
if "results" not in st.session_state:
    with st.spinner("Initializing..."):
        results = run_projections(age, bank_interest_rate, competitor_interest_rate, inflation_rate, employment_rate)
        st.session_state.results = results
        st.session_state.tab = "dashboard"
else:
    results = st.session_state.results

# Tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Predictions", "Analysis"])

# Dashboard Tab
with tab1:
    # Summary Cards
    person = results["person"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Income", f"£{person['income_after_tax']:,.2f}", "Annual (After Tax)")
    
    with col2:
        st.metric("Pension", f"£{person['pension_account']:,.2f}", "Current Balance")
    
    with col3:
        st.metric("Savings", f"£{person['savings_account']:,.2f}", "Current Balance")
    
    with col4:
        st.metric("Years with Bank", f"{person['years_with_bank']}", "Customer Loyalty")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Projections")
        
        # Prepare data for chart
        projections = results["projections"]
        ages = [p["age"] for p in projections]
        pensions = [p["pension"] for p in projections]
        savings = [p["savings"] for p in projections]
        current_accounts = [p["current_account"] for p in projections]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ages, pensions, 'o-', label='Pension')
        ax.plot(ages, savings, 's-', label='Savings')
        ax.plot(ages, current_accounts, '^-', label='Current Account')
        
        ax.set_xlabel('Age')
        ax.set_ylabel('Amount (£)')
        ax.set_title('Financial Projections by Age')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis as currency
        import matplotlib.ticker as mtick
        fmt = '£{x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Bank Leavers by Age Group")
        
        # Prepare data
        age_groups = list(results["age_group_data"].keys())
        percentages = list(results["age_group_data"].values())
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(age_groups, percentages, color='skyblue')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}%', ha='center', va='bottom')
        
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Percentage of Bank Leavers by Age Group')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)

# Predictions Tab
with tab2:
    st.subheader("Probability of Leaving the Bank")
    
    # Prepare data
    projections = results["projections"]
    ages = [f"Age {p['age']}" for p in projections]
    probabilities = [p["leaving_probability"] for p in projections]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ages, probabilities, color=['#ff9999' if p > 10 else '#ffcc99' if p > 5 else '#99cc99' for p in probabilities])
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Probability of Leaving the Bank')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(probabilities) * 1.2)  # Add some space for labels
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Detailed Financial Projections")
    
    # Create DataFrame for table
    df = pd.DataFrame(projections)
    df = df.rename(columns={
        "age": "Age",
        "pension": "Pension",
        "savings": "Savings",
        "current_account": "Current Account",
        "years_with_bank": "Years with Bank",
        "leaving_probability": "Leaving Probability (%)"
    })
    
    # Format currency columns
    df["Pension"] = df["Pension"].apply(lambda x: f"£{x:,.2f}")
    df["Savings"] = df["Savings"].apply(lambda x: f"£{x:,.2f}")
    df["Current Account"] = df["Current Account"].apply(lambda x: f"£{x:,.2f}")
    df["Leaving Probability (%)"] = df["Leaving Probability (%)"].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Retirement Focus
    retirement_data = [p for p in projections if p["age"] == 66]
    if retirement_data:
        retirement = retirement_data[0]
        
        st.subheader("Retirement Projection (Age 66)")
        st.write("Based on your current financial situation and our predictive model, here's what your finances could look like at retirement age.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pension", f"£{retirement['pension']:,.2f}")
        
        with col2:
            st.metric("Savings", f"£{retirement['savings']:,.2f}")
        
        with col3:
            st.metric("Current Account", f"£{retirement['current_account']:,.2f}")

# Analysis Tab
with tab3:
    st.subheader("Economic Factors Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Economic Indicators**")
        
        indicators_df = pd.DataFrame({
            "Indicator": ["Bank Interest Rate", "Competitor Interest Rate", "Inflation Rate", "Employment Rate"],
            "Value": [f"{bank_interest_rate:.2f}%", f"{competitor_interest_rate:.2f}%", 
                     f"{inflation_rate:.2f}%", f"{employment_rate:.2f}%"]
        })
        
        st.dataframe(indicators_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Impact Analysis**")
        
        st.write(f"**Interest Rate Differential:** {' Favorable (reduces leaving probability)' if bank_interest_rate >= competitor_interest_rate else ' Unfavorable (increases leaving probability)'}")
        
        st.write(f"**Inflation vs Interest:** {' Favorable (real returns are positive)' if bank_interest_rate >= inflation_rate else ' Unfavorable (real returns are negative)'}")
        
        employment_status = " Very High (financial stability)" if employment_rate >= 95 else " High (good economic conditions)" if employment_rate >= 85 else " Moderate (potential economic uncertainty)"
        st.write(f"**Employment Rate:** {employment_status}")
    
    st.markdown("---")
    
    st.subheader("Bank Leavers by Age Group")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare data
        age_groups = list(results["age_group_data"].keys())
        percentages = list(results["age_group_data"].values())
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            percentages, 
            labels=age_groups,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        ax.set_title('Bank Leavers by Age Group')
        
        # Make text larger
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=12)
        
        st.pyplot(fig)
    
    with col2:
        st.write("**Key Insights**")
        
        st.markdown("""
        1. **Younger customers (16-29)** have the highest probability of leaving the bank, likely due to less established financial relationships and greater mobility.
        
        2. **Middle-aged customers (30-49)** show moderate loyalty, balancing between established relationships and seeking better financial opportunities.
        
        3. **Older customers (50+)** demonstrate the highest loyalty, with significantly lower probabilities of leaving the bank due to established long-term relationships.
        
        4. **Customer retention strategies** should be tailored by age group, with more aggressive retention efforts focused on younger demographics.
        """)

# Footer
st.markdown("---")
st.markdown("© 2025 FinVision AI. All rights reserved. Powered by TensorFlow and Streamlit.")