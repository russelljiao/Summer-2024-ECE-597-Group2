import email
from email import policy
import pandas as pd

def parse1():
    df = pd.read_csv('/Users/yilu/Downloads/pythonProject/emails.csv', encoding='utf-8')
    df.drop('file', axis=1, inplace=True)
    subjects = list()
    bodies = list()
    for _, row in df.iterrows():
        subject, body = t(row['message'])
        subjects.append(subject)
        bodies.append(body)
    df.drop('message', axis=1, inplace=True)
    df['Subject'] = subjects
    df['Body'] = bodies
    df['Label'] = 0
    return df

def parse2():
    df = pd.read_csv('/Users/yilu/Downloads/pythonProject/CaptstoneProjectData_2024.csv', encoding='utf-8')
    # Remove the column at index 3
    df.drop(df.columns[[2, 3]], axis=1, inplace=True)
    df['Label'] = 1
    return df

def t(email_source):
    msg = email.message_from_string(email_source, policy=policy.default)
    subject = msg['subject']
    temp = msg.get_content_charset()
    if temp is None:
        temp = "utf-8"
    body = msg.get_payload(decode=True).decode(temp)
    return subject, body


if __name__ == '__main__':
    # df1 = parse1()
    df1 = parse1()
    df2 = parse2()
    print(df1.info())
    print(df2.info())
    merged_df = pd.concat([df1, df2], axis=0)
    print(merged_df.info())
    print(merged_df.head())
    # Reset the index of the merged DataFrame
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.to_csv('/Users/yilu/Downloads/pythonProject/Conmbined_email.csv', index=False, encoding='utf-8')
