import random
# Dictionary to store agents
agentdict = {}

filename = f"client_data.py"

for i in range(1000):  # Generate 1000 agents
    age = random.randint(16, 90)
    years_with_bank = max(1, age - 15)  # Ensuring valid range

    # Default values
    savings_rate = 0  
    pension_contribution_rate = 0  
    pension_account = 0  
    current_account = 0  
    left_bank = 0

    # Generating income based on age group
    if 16 <= age <= 17:
        income_before_tax = random.randint(0, 10910)
        savings_rate = 0.10
    elif 18 <= age <= 21:
        income_before_tax = random.randint(10910, 17284)
        savings_rate = 0.14
    elif 22 <= age <= 29:
        income_before_tax = random.randint(17284, 24600)
        savings_rate = 0.20
        pension_contribution_rate = random.uniform(0.05, 0.10)
    elif 30 <= age <= 39:
        income_before_tax = random.randint(24600, 30865)
        savings_rate = 0.24
        pension_contribution_rate = random.uniform(0.15, 0.25)
    elif 40 <= age <= 49:
        income_before_tax = random.randint(30865, 33477)
        savings_rate = 0.28
        pension_contribution_rate = random.uniform(0.20, 0.30)
    elif 50 <= age <= 59:
        income_before_tax = random.randint(31358, 33477)
        savings_rate = 0.30
        pension_contribution_rate = random.uniform(0.30, 0.40)
    elif age >= 60:
        income_before_tax = random.randint(27508, 31358)
        savings_rate = 0.32
        pension_contribution_rate = 0  # No more contributions, only withdrawals

    # Calculate tax
    if income_before_tax <= 12500:
        tax_rate = 0
    elif income_before_tax <= 25000:
        tax_rate = 0.2
    elif income_before_tax <= 50000:
        tax_rate = 0.4
    else:
        tax_rate = 0.45

    # Calculate income after tax
    income_after_tax = (1 - tax_rate) * income_before_tax

    # Expenses (50-80% of after-tax income)
    expense_rate = random.uniform(0.50, 0.80)
    yearly_expenses = income_after_tax * expense_rate

    # Yearly savings and total savings
    yearly_savings = income_after_tax * savings_rate
    savings_account = yearly_savings * years_with_bank

    # Pension Contributions & Growth
    if pension_contribution_rate > 0:
        annual_pension_contribution = income_after_tax * pension_contribution_rate
        for year in range(years_with_bank):
            pension_account = (pension_account + annual_pension_contribution) * 1.05  # 5% yearly growth

    # Current Account Calculation
    total_income_over_time = (income_after_tax - yearly_expenses) * years_with_bank  # Money earned & kept over time
    current_account = total_income_over_time - savings_account  

     #chance of leaving bank 
    if age <= 29 and random.uniform(0, 1) < 0.05:  
        left_bank = 1 #1 is yes, 0 is no 
    elif 30 <= age <= 59 and random.uniform(0, 1) < 0.2:  
        left_bank = 1
    elif age >= 60 and random.uniform(0, 1) < 0.4:  
        left_bank = 1

    # Store in dictionary
    agentdict[i] = {
        "age": age,
        "income_after_tax": round(income_after_tax, 2),
        "years_with_bank": years_with_bank,
        "savings_account": round(savings_account, 2),
        "current_account": round(current_account, 2), 
        "pension_account": round(pension_account, 2) if pension_account > 0 else 0,
        "left_bank": left_bank,
    }

with open(filename, "w") as file:
    file.write("agentdict = ")
    file.write(repr(agentdict))



