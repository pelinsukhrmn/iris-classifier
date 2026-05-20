# My first machine learning project!!
# I learned about the Iris dataset from a YouTube tutorial and decided to try it myself
# The goal is to predict what type of flower it is based on measurements

# first i need to import the stuff i need
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load the dataset (this is a built-in dataset in sklearn, very convenient!)
print("Loading the Iris dataset...")
iris = load_iris()

# the data is the measurements (sepal length, sepal width, petal length, petal width)
X = iris.data
# the target is the flower type (0=setosa, 1=versicolor, 2=virginica)
y = iris.target

print("Dataset loaded!")
print(f"Total number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Flower types: {iris.target_names}")

# split into training and testing sets
# I used 80% for training and 20% for testing (I saw this in the tutorial)
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# create the model - I chose Decision Tree because it's easy to understand
print("\nTraining the model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("Model trained!")

# now let's see how well it does on the test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# let's also try predicting a single flower manually
# these are made-up measurements
sample_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample_flower)
flower_name = iris.target_names[prediction[0]]
print(f"\nSample prediction:")
print(f"Measurements: {sample_flower[0]}")
print(f"Predicted flower: {flower_name}")

print("\nDone! My first ML model works :)")
