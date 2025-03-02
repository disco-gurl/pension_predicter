import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from churn import age, simulated_person, counter
import graphs

# Set page config
st.set_page_config(
    page_title="AI Financial Predictor",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("AI-powered Financial Predictor")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Simulated Customer Profile")
    
    # Create a styled container for the profile
    with st.container():
        # Get current person's data
        current = simulated_person[0]
        age = current['age']
        income = current['income_after_tax']
        savings = current['savings_account']
        pension = current['pension_account']
        likelihood = current['likelihood_to_leave']
        
        # Display basic information
        st.metric("Age", f"{age} years")
        st.metric("Annual Income (After Tax)", f"£{income:,.2f}")
        st.metric("Savings Account", f"£{savings:,.2f}")
        st.metric("Pension Account", f"£{pension:,.2f}")
        
        # Risk Assessment
        color = "green" if likelihood < 30 else "orange" if likelihood < 70 else "red"
        risk_level = "Low Risk" if likelihood < 30 else "Medium Risk" if likelihood < 70 else "High Risk"
            
        st.markdown(f"""
        <div style='background-color: rgba(0,0,0,0.1); padding: 20px; border-radius: 10px;'>
            <h3 style='color: {color}; margin: 0;'>Likelihood to Leave: {likelihood:.2f}%</h3>
            <p style='color: {color}; margin: 5px 0 0 0;'>{risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pie chart
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        sizes = [100 - counter, counter]
        colors = ['#2ecc71' if counter < 30 else '#e67e22' if counter < 70 else '#e74c3c', '#bdc3c7']
        
        ax_pie.pie(sizes, 
                  labels=['Stay', 'Leave'],
                  colors=colors,
                  autopct='%1.1f%%',
                  startangle=90)
        ax_pie.axis('equal')
        plt.title(f"Age Group {age} Distribution")
        st.pyplot(fig_pie)
        plt.clf()

with col2:
    st.subheader("Financial Projections")
    tabs = st.tabs(["+1 year", "+5 years", "+10 years"])
    
    # Projections
    for i, tab in enumerate(tabs, 1):
        with tab:
            person = simulated_person[i]
            age = person['age']
            income = person['income_after_tax']
            savings = person['savings_account']
            pension = person['pension_account']
            
            st.metric("Age", f"{age} years")
            st.metric("Projected Annual Income", f"£{income:,.2f}")
            st.metric("Projected Pension", f"£{pension:,.2f}")
            st.metric("Projected Savings", f"£{savings:,.2f}")

st.markdown("---")

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

st.markdown("---")
st.markdown("""
<small>Disclaimer: This tool provides estimates based on historical data and current market conditions.</small>
""", unsafe_allow_html=True)
