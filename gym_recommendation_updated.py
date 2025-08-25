import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
import warnings
import pickle
import os
import numpy as np

warnings.filterwarnings("ignore")

# Step 1: Load Dataset
df1 = pd.read_csv(r"gym_recommendation_with_health.csv")  # Update path if needed
df = df1.copy()
print("Columns:", df.columns.tolist())
print(df.head())
print("Shape:", df.shape)

# Step 2: Dataset Visualization (Plots will show; close to continue)
# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.show()

# Fitness Goal count
plt.figure(figsize=(6,4))
sns.countplot(x='Fitness Goal', data=df, palette='viridis')
plt.title("Distribution of Fitness Goals")
plt.xticks(rotation=45)
plt.show()

# Fitness Type by Level
plt.figure(figsize=(8,5))
cross_tab = pd.crosstab(df['Level'], df['Fitness Type'])
cross_tab.plot(kind='bar', stacked=True, colormap='Paired')
plt.title("Fitness Type by Level")
plt.show()

# BMI vs Age by Fitness Goal
plt.figure(figsize=(7,5))
sns.scatterplot(x='Age', y='BMI', hue='Fitness Goal', data=df, palette='Set2')
plt.title("BMI vs Age by Fitness Goal")
plt.show()

# Correlation heatmap (include new numeric column)
plt.figure(figsize=(10,6))
sns.heatmap(df[['Age','Height','Weight','BMI','WaistCircumference_in']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Top 5 most common exercises
exercise_counts = df['Exercises'].value_counts().head(5)
plt.figure(figsize=(7,4))
plt.barh(exercise_counts.index, exercise_counts.values, color='orange')
plt.title("Top 5 Most Common Exercises")
plt.show()

# Step 3: Drop Irrelevant Columns
df.drop(columns=['ID'], axis=1, inplace=True)
print(df.head())

# Step 4: Encode Categorical Columns
input_label_encoders = {}
input_cols = ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type', 'HealthCondition']  # Added HealthCondition
for col in input_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    input_label_encoders[col] = le

output_label_encoders = {}
output_cols = ['Exercises', 'Equipment', 'Diet', 'Recommendation']
for col in output_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    output_label_encoders[col] = le

print(df.head())

# Step 5: Define Features and Outputs
X = df[['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type', 'WaistCircumference_in', 'HealthCondition']]  # Added new columns
y = df[output_cols]

# Step 6: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nx_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Step 7: Scale Features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 8: Define and Train Multiple Models
clf1 = LogisticRegression(solver='liblinear', random_state=42)
clf2 = SVC(random_state=42, probability=True)
clf3 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf4 = KNeighborsClassifier()
clf5 = DecisionTreeClassifier(max_leaf_nodes=20, random_state=42)
clf6 = GaussianNB()
clf7 = AdaBoostClassifier(random_state=42)
clf8 = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)  # Optimized to avoid long training

models = {
    "Logistic Regression": clf1,
    "Support Vector Classifier": clf2,
    "Random Forest Classifier": clf3,
    "K Nearest Neighbors": clf4,
    "Decision Tree Classifier": clf5,
    "Gaussian Naive Bayes": clf6,
    "AdaBoost Classifier": clf7,
    "Gradient Boosting Classifier": clf8,
    "SVC_RandomForest_Voting": VotingClassifier(estimators=[('svc', clf2), ('rf', clf3)], voting='soft')
}

for target in output_cols:
    print("\n" + "="*80)
    print(f"Training models for target: {target}")
    print("="*80)
    y_train_target = y_train[target]
    y_test_target = y_test[target]
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(x_train, y_train_target)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test_target, y_pred)
        print(f"Accuracy: {acc}")
        print("Classification Report:\n", classification_report(y_test_target, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test_target, y_pred))

# Step 9: Model Accuracy Comparison
accuracy_results = {target: {} for target in output_cols}
for target in output_cols:
    y_train_target = y_train[target]
    y_test_target = y_test[target]
    for name, clf in models.items():
        clf.fit(x_train, y_train_target)
        y_pred = clf.predict(x_test)
        accuracy_results[target][name] = accuracy_score(y_test_target, y_pred)

acc_df = pd.DataFrame(accuracy_results)
plt.figure(figsize=(12,6))
sns.heatmap(acc_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Model Accuracy Comparison for Each Output")
plt.show()

# Step 10: Select and Train Multi-Output Model (Random Forest)
multi_model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
multi_model.fit(x_train_scaled, y_train)
y_pred = multi_model.predict(x_test_scaled)
y_pred_df = pd.DataFrame(y_pred, columns=output_cols)

for target in output_cols:
    print("\n" + "="*80)
    print(f"Evaluation for target: {target}")
    print("="*80)
    y_test_target = y_test[target]
    y_pred_target = y_pred_df[target]
    acc = accuracy_score(y_test_target, y_pred_target)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test_target, y_pred_target))
    print("Confusion Matrix:\n", confusion_matrix(y_test_target, y_pred_target))

# Step 11: Display Sample Predictions
for i in range(5):
    pred = multi_model.predict(x_test_scaled[i].reshape(1, -1))[0]
    actual = y_test.iloc[i].values
    print(f"Sample {i}: Predicted = {pred}, Actual = {actual}")

# Step 12: Save Files
os.makedirs("Models", exist_ok=True)
pickle.dump(scaler, open("Models/scaler.pkl", "wb"))
pickle.dump(multi_model, open("Models/model.pkl", "wb"))
pickle.dump(output_label_encoders, open("Models/output_label_encoders.pkl", "wb"))
pickle.dump(input_label_encoders, open("Models/input_label_encoders.pkl", "wb"))

# Step 13: Load Files (for testing)
scaler = pickle.load(open("Models/scaler.pkl", 'rb'))
model = pickle.load(open("Models/model.pkl", 'rb'))
output_label_encoders = pickle.load(open("Models/output_label_encoders.pkl", 'rb'))
input_label_encoders = pickle.load(open("Models/input_label_encoders.pkl", 'rb'))

# Step 14: Recommendation System Function
def safe_encode(encoder, value, col_name):
    if value not in encoder.classes_:
        raise ValueError(f"Value '{value}' not found for '{col_name}'. Allowed: {list(encoder.classes_)}")
    return encoder.transform([value])[0]

def Recommendations(sex, age, height, weight, hypertension, diabetes, bmi, level, fitness_goal, fitness_type, waist_circumference, health_condition):
    sex_encoded = safe_encode(input_label_encoders['Sex'], sex, 'Sex')
    hypertension_encoded = safe_encode(input_label_encoders['Hypertension'], hypertension, 'Hypertension')
    diabetes_encoded = safe_encode(input_label_encoders['Diabetes'], diabetes, 'Diabetes')
    level_encoded = safe_encode(input_label_encoders['Level'], level, 'Level')
    goal_encoded = safe_encode(input_label_encoders['Fitness Goal'], fitness_goal, 'Fitness Goal')
    type_encoded = safe_encode(input_label_encoders['Fitness Type'], fitness_type, 'Fitness Type')
    health_encoded = safe_encode(input_label_encoders['HealthCondition'], health_condition, 'HealthCondition')
    
    features = np.array([[sex_encoded, age, height, weight, hypertension_encoded, diabetes_encoded, bmi, level_encoded, goal_encoded, type_encoded, waist_circumference, health_encoded]])
    scaled_features = scaler.transform(features)
    encoded_preds = model.predict(scaled_features)[0]
    
    decoded_preds = {
        col: output_label_encoders[col].inverse_transform([encoded_preds[i]])[0]
        for i, col in enumerate(output_cols)
    }
    return decoded_preds

# Step 15: Test Recommendation
try:
    result = Recommendations(
        sex="Male", age=28, height=175, weight=72, hypertension="No", diabetes="No",
        bmi=23.5, level="Normal", fitness_goal="Weight Loss", fitness_type="Cardio Fitness",
        waist_circumference=23.5, health_condition="None"  # Adjust based on your data
    )
    print("Exercises:", result['Exercises'])
    print("Equipment:", result['Equipment'])
    print("Diet:", result['Diet'])
    print("Recommendation:", result['Recommendation'])
except ValueError as e:
    print(e)