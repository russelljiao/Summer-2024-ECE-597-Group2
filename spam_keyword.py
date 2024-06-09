import pandas as pd
import re
from collections import Counter

# Load data
file_path = '/Users/zhangguoyu/Downloads/CaptstoneProjectData_2024.csv'
data = pd.read_csv(file_path)

# Custom stop words list
stop_words = set([
    "I", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
])

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(tokens)

# Function to extract basic features
def extract_basic_features(df):
    df['subject_length'] = df['Subject'].apply(lambda x: len(x))
    df['body_length'] = df['Body'].apply(lambda x: len(x))
    df['subject_word_count'] = df['Subject'].apply(lambda x: len(x.split()))
    df['body_word_count'] = df['Body'].apply(lambda x: len(x.split()))
    df['subject_unique_word_count'] = df['Subject'].apply(lambda x: len(set(x.split())))
    df['body_unique_word_count'] = df['Body'].apply(lambda x: len(set(x.split())))
    df['subject_special_char_count'] = df['Subject'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))
    df['body_special_char_count'] = df['Body'].apply(lambda x: len(re.findall(r'[^\w\s]', x)))
    return df

# Define common spam keywords
spam_keywords = ["free", "win", "winner", "prize", "click here", "limited time", "offer", "buy now", "congratulations", "urgent", "money", "cash", "reward", "claim", "gift"]

# Function to count keyword occurrences
def count_spam_keywords(text, keywords):
    text = text.lower()
    return sum(text.count(keyword) for keyword in keywords)

# Extract keyword features
def extract_spam_keyword_features(df, keywords):
    for keyword in keywords:
        df[f'subject_keyword_{keyword.replace(" ", "_")}'] = df['Subject'].apply(lambda x: x.count(keyword))
        df[f'body_keyword_{keyword.replace(" ", "_")}'] = df['Body'].apply(lambda x: x.count(keyword))
    return df

# Handle missing values, replace NaN with empty string
data['Subject'] = data['Subject'].fillna('')
data['Body'] = data['Body'].fillna('')

# Preprocess text
data['Subject'] = data['Subject'].apply(preprocess_text)
data['Body'] = data['Body'].apply(preprocess_text)

# Extract basic features
data = extract_basic_features(data)

# Extract spam keyword features
data = extract_spam_keyword_features(data, spam_keywords)

# Save the processed data to file
output_file_path = '/Users/zhangguoyu/Downloads/feature.csv'
data.to_csv(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")