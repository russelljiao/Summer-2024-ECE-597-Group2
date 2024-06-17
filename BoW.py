import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

file_path = '/Users/yilu/Desktop/pythonProject/CaptstoneProjectData_2024.csv'
data = pd.read_csv(file_path)

data['Subject'] = data['Subject'].fillna('')
data['Body'] = data['Body'].fillna('')

def simple_preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    text = text.replace('________________________________', '')
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

data['Cleaned_Subject'] = data['Subject'].apply(simple_preprocess_text)
data['Cleaned_Body'] = data['Body'].apply(simple_preprocess_text)

data['Cleaned_Text'] = data['Cleaned_Subject'] + " " + data['Cleaned_Body']

print("Cleaned data:")
print(data[['Cleaned_Subject', 'Cleaned_Body', 'Cleaned_Text']].head())


count_vectorizer = CountVectorizer(max_features=500)  # Limit to 500 features
bow_matrix = count_vectorizer.fit_transform(data['Cleaned_Text'])
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())


print("BoW feature matrix:")
print(bow_df.head())

output_file_path = '/Users/yilu/Desktop/pythonProject/feature.csv'
bow_df.to_csv(output_file_path, index=False)

print(f"BoW features saved to file: {output_file_path}")