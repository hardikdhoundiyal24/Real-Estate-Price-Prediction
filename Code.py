# Real Estate Price Prediction in Bangalore

# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loading and Exploring the Data
data = pd.read_csv("bengaluru_house_prices.csv")
print(data.head())
print(data.info())
print(data.describe())

# Data Preprocessing
# Dropping unnecessary columns
data = data.drop(['area_type', 'availability', 'society', 'balcony'], axis=1)

# Handling missing values
data = data.dropna()

# Converting 'size' to numerical (e.g., "2 BHK" to 2)
data['bhk'] = data['size'].apply(lambda x: int(x.split(' ')[0]))
data = data.drop(['size'], axis=1)

# Cleaning 'total_sqft' column (handling ranges like "1133 - 1384")
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
data = data.dropna()

# Encoding categorical variable 'location'
data['location'] = data['location'].apply(lambda x: x.strip())
location_stats = data['location'].value_counts()
location_stats_less_than_10 = location_stats[location_stats <= 10]
data['location'] = data['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Feature Engineering
# Adding price per square foot
data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']

# Removing outliers based on price per square foot
data = data[(data['price_per_sqft'] >= 3000) & (data['price_per_sqft'] <= 15000)]

# Preparing features and target variable
X = data.drop(['price', 'price_per_sqft'], axis=1)
y = data['price']

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Making a Prediction
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == f'location_{location}')[0][0] if f'location_{location}' in X.columns else -1
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

# Example prediction
print(predict_price('Indira Nagar', 1000, 2, 2))
