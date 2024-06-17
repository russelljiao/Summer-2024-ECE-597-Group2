import pandas as pd

# Load the processed phishing and normal mail datasets
phishing_path = '/Users/jiaoyihan/capstone/capstone_project/Processed_CaptstoneProjectData_2024_ngram.csv'
normal_path = '/Users/jiaoyihan/capstone/capstone_project/Processed_emails_ngram.csv'

phishing_data = pd.read_csv(phishing_path)
normal_data = pd.read_csv(normal_path)

# Add label columns: 1 for phishing emails and 0 for normal emails
phishing_data['label'] = 1
normal_data['label'] = 0

# Merge data set
combined_data = pd.concat([phishing_data, normal_data], ignore_index=True)

# Save the merged data set
output_path = '/Users/jiaoyihan/capstone/capstone_project/Combined_emails_ngram.csv'
combined_data.to_csv(output_path, index=False)

print(f"The combined data set is saved to a file: {output_path}")
