import pandas as pd

# Load the dataset
file_path = 'emails.csv' 
emails = pd.read_csv(file_path)

print(emails.head())

print(emails.info())

print(emails.isnull().sum())

# Convert the 'message' column to lowercase
emails['message'] = emails['message'].str.lower()

# Verify the changes
print(emails['message'].head())

emails.dropna(subset=['file', 'message'], inplace=True)

def is_spam(email_subject):
    spam_keywords = ['credit', 'promo', 'promotion', 'debit']
    return any(keyword in email_subject.lower() for keyword in spam_keywords)

# Apply the function to filter out spam emails
emails = emails[~emails['subject'].apply(is_spam)]

# Further clean up by removing system auto-generated emails, for example
# system_senders = ['sample1@enron.com', 'sample2@enron.com']
# emails = emails[~emails['from'].isin(system_senders)]
