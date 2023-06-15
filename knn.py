from sklearn.neighbors import KNeighborsClassifier

# Prepare the data
X = df[['sepalWidth', 'petalWidth']]
y = df['irisSpecies']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = knn_model.score(X_test, y_test)
print("Accuracy:", accuracy)
