import pandas as pd
from backend.agents.data_scientist import DataScientistAgent
import json
import os

# Create sample CSV data (10 rows, 5 columns)
data = {
    'id': range(1, 11),
    'age': [25, 30, 35, 40, 45, 50, 55, 60, None, 70],
    'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'score': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
}
df = pd.DataFrame(data)

os.makedirs('artifacts', exist_ok=True)
csv_path = 'artifacts/test_data.csv'
df.to_csv(csv_path, index=False)

# Instantiate the agent
print("Initializing DataScientistAgent...")
agent = DataScientistAgent()

print(f"Calling analyze() on {csv_path}...")
result = agent.analyze([csv_path], 'Predict customer score based on age and income.')

print("\n--- Output ---")
# Print the result nicely formatted
if 'error' in result:
    print(f"Error: {result['error']}")
else:
    for key, value in result.items():
        if key == 'eda_summary':
            print(f"EDA Summary: {json.dumps(value, indent=2)}")
        else:
            print(f"{key}: {value}")
            
print("\nTest completed.")
