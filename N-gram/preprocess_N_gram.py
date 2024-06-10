import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# 加载数据集
file_path = '/Users/jiaoyihan/capstone/capstone_project/CaptstoneProjectData_2024.csv'
data = pd.read_csv(file_path)

# 填充空值
data['Subject'] = data['Subject'].fillna('')
data['Body'] = data['Body'].fillna('')

# 简单的预处理函数
def simple_preprocess_text(text):
    # 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 移除URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 移除特殊字符和数字
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # 转换为小写
    text = text.lower()
    # 移除连续的下划线
    text = text.replace('________________________________', '')
    # 分词并移除停用词
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

# 应用预处理函数到Subject和Body列
data['Cleaned_Subject'] = data['Subject'].apply(simple_preprocess_text)
data['Cleaned_Body'] = data['Body'].apply(simple_preprocess_text)

# 合并清理后的Subject和Body文本
data['Cleaned_Text'] = data['Cleaned_Subject'] + " " + data['Cleaned_Body']

# 检查清理后的数据
print("Data after process:")
print(data[['Cleaned_Subject', 'Cleaned_Body', 'Cleaned_Text']].head())

# 初始化 N-gram Vectorizer
ngram_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=500)  # 限制最多500个特征

# 拟合并转换清理后的文本数据
ngram_matrix = ngram_vectorizer.fit_transform(data['Cleaned_Text'])

# 将 N-gram 矩阵转换为 DataFrame 以便于可视化
ngram_df = pd.DataFrame(ngram_matrix.toarray(), columns=ngram_vectorizer.get_feature_names_out())

# 检查 N-gram DataFrame
print("N-gram characteristic matrix:")
print(ngram_df.head())

# 保存 N-gram 特征到 CSV 文件，不带索引
output_file_path = '/Users/jiaoyihan/capstone/capstone_project/Processed_CaptstoneProjectData_2024_ngram.csv'
ngram_df.to_csv(output_file_path, index=False)

print(f"N-gram feature has been stored to new csv: {output_file_path}")
