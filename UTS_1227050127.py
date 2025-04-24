# %%
# Load Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Load dataset
df = pd.read_csv('dataset_buys _comp.csv')

# %%
# Eksplorasi awal
print(df.head())
print(df.info())
print(df.isnull().sum())

# %%
# Pra-pemrosesan: Label Encoding untuk semua kolom kategorikal
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# %%
# Split training dan testing data
x = df.drop('Buys_Computer', axis=1)
y = df['Buys_Computer']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# %%
# Buat dan latih model Decision Tree
model = DecisionTreeClassifier(random_state=10)
model.fit(x_train, y_train)

# %%
# Evaluasi
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %% 
# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# %%
# Visualisasi pohon keputusan
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=x.columns, class_names=["No", "Yes"], filled=True)
plt.title("Decision Tree")
plt.show()