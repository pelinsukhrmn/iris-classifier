from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("Loading the Iris dataset...")
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded!")
print(f"Total number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Flower types: {iris.target_names}")

print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nTraining the model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("Model trained!")

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

sample_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample_flower)
flower_name = iris.target_names[prediction[0]]
print(f"\nSample prediction:")
print(f"Measurements: {sample_flower[0]}")
print(f"Predicted flower: {flower_name}")

print("\nDone! My first ML model works :)")
