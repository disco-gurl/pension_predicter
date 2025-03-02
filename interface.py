import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import tensorflow as tf
import random
from churn import (
    get_pension_savings_rate, simulations, baseInflation, 
    decision, counter, age, simulated_person
)
import graphs

# Initialize session state for counter if it doesn't exist
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'inflation_history' not in st.session_state:
    st.session_state.inflation_history = []

# Set page config
st.set_page_config(
    page_title="AI Financial Predictor",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("AI-powered Financial Predictor")
st.markdown("""
This application helps predict and visualize your financial future using AI and machine learning.
Enter your details below to get started!
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    # Display the age from churn.py
    st.info(f"Current Age: {age}")
    
    # Economic indicators
    st.subheader("Economic Indicators")
    bank_rate = st.slider("Bank Interest Rate (%)", 0.0, 10.0, 2.0)
    competitor_rate = st.slider("Competitor Interest Rate (%)", 0.0, 10.0, 1.8)
    
    # Calculate dynamic inflation based on age
    if st.button("Recalculate Inflation"):
        st.session_state.counter += 1
        current_inflation = baseInflation
        st.session_state.inflation_history = [baseInflation]
        
        for i in range(age):
            if age < 66:
                randomNum = random.randrange(1, 4)
                randomDec = random.choice(decision)
                current_inflation += randomNum if randomDec else -randomNum
                st.session_state.inflation_history.append(current_inflation)
        
        inflation = current_inflation
    else:
        inflation = st.slider("Inflation Rate (%)", 0.0, 10.0, 2.12)
    
    employment = st.slider("Employment Rate (%)", 0.0, 100.0, 80.0)
    
    # Display counter and inflation history
    st.subheader("Simulation Statistics")
    st.write(f"Number of Recalculations: {st.session_state.counter}")
    
    if st.session_state.inflation_history:
        # Create a line chart for inflation history
        st.subheader("Inflation Rate History")
        fig_inflation = plt.figure(figsize=(8, 4))
        plt.plot(st.session_state.inflation_history, marker='o')
        plt.title("Inflation Rate Changes Over Time")
        plt.xlabel("Years")
        plt.ylabel("Inflation Rate (%)")
        plt.grid(True)
        st.pyplot(fig_inflation)
        plt.clf()

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Simulated Customer Profile")
    
    # Display simulated person's information in a nice format
    st.markdown("""
    #### Personal Information
    """)
    
    # Create a styled container for the profile
    with st.container():
        # Basic information
        st.metric("Age", f"{age} years")
        st.metric("Annual Income (After Tax)", f"£{simulated_person['income_after_tax']:,.2f}")
        
        # Financial accounts
        st.markdown("#### Financial Accounts")
        col_savings, col_pension = st.columns(2)
        
        with col_savings:
            st.metric("Savings Account", f"£{simulated_person['savings_account']:,.2f}")
        with col_pension:
            st.metric("Pension Account", f"£{simulated_person['pension_account']:,.2f}")
        
        # Likelihood to leave
        st.markdown("#### Risk Assessment")
        likelihood = simulated_person['likehood_to_leave']
        
        # Color code based on likelihood
        if likelihood < 30:
            color = "green"
            risk_level = "Low Risk"
        elif likelihood < 70:
            color = "orange"
            risk_level = "Medium Risk"
        else:
            color = "red"
            risk_level = "High Risk"
            
        st.markdown(f"""
        <div style='background-color: rgba(0,0,0,0.1); padding: 20px; border-radius: 10px;'>
            <h3 style='color: {color}; margin: 0;'>Likelihood to Leave: {likelihood:.2f}%</h3>
            <p style='color: {color}; margin: 5px 0 0 0;'>{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add pie chart for counter
        st.markdown("#### Age Group Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        
        # Data for pie chart
        labels = ['Stay', 'Leave']
        sizes = [100 - likelihood, likelihood]  # Using likelihood from simulated_person
        colors = ['#2ecc71' if likelihood < 30 else '#e67e22' if likelihood < 70 else '#e74c3c', '#bdc3c7']
        explode = (0.1, 0)  # explode the 'Leave' slice
        
        # Create pie chart
        wedges, texts, autotexts = ax_pie.pie(sizes, 
                                             explode=explode,
                                             labels=labels,
                                             colors=colors,
                                             autopct='%1.1f%%',
                                             shadow=True,
                                             startangle=90)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax_pie.axis('equal')
        
        # Style the text and percentages
        plt.setp(autotexts, size=12, weight="bold")
        plt.setp(texts, size=12, weight="bold")
        
        # Add title
        plt.title(f"Customer Distribution", pad=20, fontsize=14, fontweight='bold')
        
        # Display the pie chart
        st.pyplot(fig_pie)
        
        # Clear the current figure to avoid overlapping with other charts
        plt.clf()

with col2:
    st.subheader("Financial Projections")
    
    # Get pension and savings rates
    pension_rate, savings_rate = get_pension_savings_rate(age)
    
    # Create a styled container for recommendations
    with st.container():
        st.markdown("#### Recommended Rates")
        col_pension_rate, col_savings_rate = st.columns(2)
        
        with col_pension_rate:
            st.info(f"Pension Rate: {pension_rate}%")
        with col_savings_rate:
            st.info(f"Savings Rate: {savings_rate}%")
    
    # Future projections
    st.markdown("#### Future Projections")
    age_offsets = [1, 5, 10]
    
    # Create tabs for different projections
    tabs = st.tabs([f"+{offset} years" for offset in age_offsets])
    
    for i, (tab, offset) in enumerate(zip(tabs, age_offsets)):
        with tab:
            future_age = age + offset
            if future_age <= 66:
                # Calculate projected values based on simulated person's data
                future_pension = simulated_person['pension_account'] * (1 + pension_rate/100) ** offset
                future_savings = simulated_person['savings_account'] * (1 + savings_rate/100) ** offset
                future_income = simulated_person['income_after_tax'] * (1 + inflation/100) ** offset
                
                # Display projections
                st.metric("Projected Annual Income", f"£{future_income:,.2f}")
                st.metric("Projected Pension", f"£{future_pension:,.2f}")
                st.metric("Projected Savings", f"£{future_savings:,.2f}")

# Visualization section
st.subheader("Customer Age Group Analysis")
fig, ax = plt.subplots(figsize=(10, 6))

# Get data from graphs module
age_groups = graphs.age_groups
have_left = graphs.have_left

# Calculate percentages
percentages = []
for group in age_groups:
    total = age_groups[group]
    left = have_left[group]
    percentage = (left / total * 100) if total > 0 else 0
    percentages.append(percentage)

# Create bar chart with better styling
plt.style.use('ggplot')
fig.patch.set_facecolor('#F0F2F6')
ax.set_facecolor('#F0F2F6')

bars = plt.bar(age_groups.keys(), percentages, color='#1f77b4')
plt.title("Percentage of Bank Leavers by Age Group", pad=20, fontsize=12, fontweight='bold')
plt.xlabel("Age Groups", fontsize=10)
plt.ylabel("Percentage of Leavers", fontsize=10)
plt.xticks(rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom',
             fontweight='bold')

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Additional information
st.markdown("""
### How to Use This Tool
1. View your simulated customer profile in the left panel
2. Check your recommended pension and savings rates
3. Explore future financial projections
4. Analyze your likelihood of leaving the bank
5. Compare with age group statistics

The predictions are based on historical data and current economic indicators.
""")

# Add footer with disclaimer
st.markdown("---")
st.markdown("""
<small>Disclaimer: This tool provides estimates based on historical data and current market conditions. 
Actual results may vary. Please consult with a financial advisor for personalized advice.</small>
""", unsafe_allow_html=True)
