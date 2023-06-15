from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Prepare the data
X = df[['sepalWidth', 'petalWidth']]
y = df['irisSpecies']

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the ANN model
ann_model = keras.Sequential([
    keras.layers.Dense(16, input_dim=2, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
ann_model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

# Evaluate the model
_, accuracy = ann_model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
