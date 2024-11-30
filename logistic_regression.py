import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = r'people-100.csv'
data = pd.read_csv(file_path)
data['Date of birth'] = pd.to_datetime(data['Date of birth'], errors='coerce')
current_year = pd.Timestamp.now().year
data['Age'] = current_year - data['Date of birth'].dt.year
data['Sex_encoded'] = data['Sex'].apply(lambda x: 1 if x == 'Male' else 0)

X = data[['Age']]
y = data['Sex_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100
print(f"Model is {accuracy}% accurate")
