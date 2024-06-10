import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = '/Users/zhangguoyu/Downloads/CaptstoneProjectData_2024.csv'
data = pd.read_csv(file_path)

# Fill missing values
data['Subject'] = data['Subject'].fillna('')
data['Body'] = data['Body'].fillna('')

# Simple preprocessing function
def simple_preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove continuous underscores
    text = text.replace('________________________________', '')
    # Tokenize and remove stop words
    words = text.split()
    stop_words = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could',
        "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for',
        'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's",
        'her', 'here', "here's", 'hers', 'herself', 'him', "himself", 'his', 'how', "how's", 'I', "I'd", "I'll", "I'm",
        "I've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'let', "let's", 'me', 'more', 'most',
        "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
        'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should',
        "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then',
        'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to',
        'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were',
        "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom',
        'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
        'yourself', 'yourselves'
    }
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing function to Subject and Body columns
data['Cleaned_Subject'] = data['Subject'].apply(simple_preprocess_text)
data['Cleaned_Body'] = data['Body'].apply(simple_preprocess_text)

# Combine cleaned Subject and Body texts
data['Cleaned_Text'] = data['Cleaned_Subject'] + " " + data['Cleaned_Body']

# Check the cleaned data
print("Cleaned data:")
print(data[['Cleaned_Subject', 'Cleaned_Body', 'Cleaned_Text']].head())

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=500)  # Limit to 500 features

# Fit and transform the cleaned text data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Cleaned_Text'])

# Convert the TF-IDF matrix to a DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Check the TF-IDF DataFrame
print("TF-IDF feature matrix:")
print(tfidf_df.head())

# Save the TF-IDF features to a CSV file, without index
output_file_path = '/Users/zhangguoyu/Downloads/tfidf.csv'
tfidf_df.to_csv(output_file_path, index=False)

print(f"TF-IDF features saved to file: {output_file_path}")