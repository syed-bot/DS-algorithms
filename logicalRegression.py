from sklearn.linear_model import LogisticRegression

# Prepare the data
X = df[['sepalWidth', 'petalWidth']]
y = df['irisSpecies']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate the model
accuracy = logistic_model.score(X_test, y_test)
print("Accuracy:", accuracy)
