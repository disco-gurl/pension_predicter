import importlib
import glob 
import matplotlib.pyplot as plt 

 # Access data from client_data dictionary 
filename = "client_data"
client_data = importlib.import_module(filename)
clientdict = client_data.agentdict

#counts how many people in age group 
age_groups = {
    "16-17": 0, "18-21": 0, "22-29": 0 , "30-39": 0 , "40-49": 0, "50-59": 0, ">=60": 0
} 
#how many people who have left in each age group
have_left = {
    "16-17": 0, "18-21": 0, "22-29": 0 , "30-39": 0 , "40-49": 0, "50-59": 0, ">=60": 0
} 

#counts how many clients are in each age group, and how many clients have left the bank from each age group 
for client in clientdict.values():
    age = client["age"]
    left_bank = client["left_bank"] #0 = no, 1 = yes 

    if age <=17:
        group = "16-17"
    elif age <=21:
        group = "18-21"
    elif age <= 29:
        group = "22-29"
    elif age <= 39:
        group = "30-39"
    elif age <= 49:
        group = "40-49"
    elif age <= 59:
        group = "50-59"
    elif age >= 60:
        group = ">=60"
     
    age_groups[group] = age_groups[group] + 1

    if left_bank == 1:
        have_left[group] = have_left[group] + 1

percentage_left = {
    group: (have_left[group] / age_groups[group])*100 for group in age_groups
}
    
plt.figure(figsize=(8, 5))
plt.bar(percentage_left.keys(), percentage_left.values(), color=['blue', 'green', 'red'])
plt.xlabel("Age Groups")
plt.ylabel("Percentage Left %")
plt.title("Percentage of People Who Left the Bank by Age Group")
plt.ylim(0, max(percentage_left.values()) + 5)  # Adjust y-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show values on top of bars
for index, value in enumerate(percentage_left.values()):
    plt.text(index, value + 1, f"{value:.1f}%", ha='center', fontsize=12)

plt.show()

