import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 加载合并后的数据集
combined_data_path = '/Users/jiaoyihan/capstone/capstone_project/Combined_emails_ngram.csv'
combined_data = pd.read_csv(combined_data_path)

# 用特定值（如平均值）填充缺失值
imputer = SimpleImputer(strategy='mean')
combined_data_imputed = imputer.fit_transform(combined_data)

# 转换为DataFrame
combined_data = pd.DataFrame(combined_data_imputed, columns=combined_data.columns)

# 划分特征和标签
X = combined_data.drop(columns=['label'])
y = combined_data['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练逻辑回归模型
model = LogisticRegression(max_iter=5000)  # 增加max_iter的值
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of logistic regression model: {accuracy}")

# 打印分类报告
print("classification report:")
print(classification_report(y_test, y_pred))

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵的热力图
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Phishing'], yticklabels=['Normal', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# 保存训练好的模型
model_path = '/Users/jiaoyihan/capstone/capstone_project/Logistic_Regression_model.pkl'
joblib.dump(model, model_path)
print(f"The logistic regression model is saved to a file: {model_path}")
