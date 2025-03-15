import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load the data
df = pd.read_csv('path_to_sentiment_evolution_output')

# Setting 'KLM' as the reference category for the 'airline' variable
df['airline'] = df['airline'].astype('category')
df['airline'] = df['airline'].cat.reorder_categories(
    ['KLM'] + [cat for cat in df['airline'].cat.categories if cat != 'KLM']
)

# Filtering the data
df = df[df["response_time_hours"] < 1]
df = df[df["response_time_minutes"] <= 60]

# Shifting the 'response_time_hours' to make all values positive
min_response_time_hours = df['response_time_hours'].min()
shifted_response_time_hours = df['response_time_hours'] - min_response_time_hours

# Normalizing the shifted 'response_time_hours' and 'sentiment_change' features
scaler = StandardScaler()
df[['shifted_response_time_hours', 'sentiment_change']] = scaler.fit_transform(df[['response_time_hours', 'sentiment_change']])

# Shifting back to the original scale
df['response_time_hours_normalized'] = df['shifted_response_time_hours'] + min_response_time_hours

# Creating a new column for 10-minute intervals
df['response_time_10_minutes'] = (df['response_time_minutes'] // 10) * 10


# Setting 'KLM' as the reference category for the 'airline' variable
df['airline'] = df['airline'].astype('category')
df['airline'] = df['airline'].cat.reorder_categories(
    ['KLM'] + [cat for cat in df['airline'].cat.categories if cat != 'KLM']
)

# Filtering the data
df = df[df["response_time_hours"] < 1]
df = df[df["response_time_minutes"] <= 60]

# Shifting the 'response_time_hours' to make all values positive
min_response_time_hours = df['response_time_hours'].min()
shifted_response_time_hours = df['response_time_hours'] - min_response_time_hours

# Normalizing the shifted 'response_time_hours' and 'sentiment_change' features
scaler = StandardScaler()
df[['shifted_response_time_hours', 'sentiment_change']] = scaler.fit_transform(df[['response_time_hours', 'sentiment_change']])

# Shifting back to the original scale
df['response_time_hours_normalized'] = df['shifted_response_time_hours'] + min_response_time_hours

# Creating a new column for 10-minute intervals
df['response_time_10_minutes'] = (df['response_time_minutes'] // 20) * 20

# Plot the box plot
plt.figure(figsize=(16, 14))
sns.boxplot(x=df['response_time_10_minutes'], y=df['sentiment_change'], hue=df['airline'])
plt.xlabel('Response Time (20-minute intervals)')
plt.ylabel('Change from initial to final user sentiment')
plt.title('Distribution of sentiment change per 20 Minutes reply speed', size=16, weight='bold')
plt.legend(title='Airline:')
plt.show()

# Running the regression model with interaction term and 'KLM' as the reference category
reg2 = smf.ols(formula='sentiment_change ~ response_time_hours_normalized * C(airline)', data=df).fit()

# Printing the summary of the regression model
print(reg2.summary())

# Preparing the plot data for simple slope plot
intercept = reg2.params['Intercept']
response_time_coef = reg2.params['response_time_hours_normalized']
airlines = df['airline'].unique()
plot_data = pd.DataFrame()
response_time_range = np.linspace(df['response_time_hours_normalized'].min(), df['response_time_hours_normalized'].max(), 100)

for airline in airlines:
    if airline == 'KLM':
        airline_intercept = intercept
        airline_slope = response_time_coef
    else:
        airline_intercept = intercept + reg2.params[f'C(airline)[T.{airline}]']
        airline_slope = response_time_coef + reg2.params[f'response_time_hours_normalized:C(airline)[T.{airline}]']
    
    plot_data = pd.concat([plot_data, pd.DataFrame({
        'response_time_hours_normalized': response_time_range,
        'sentiment_change': airline_intercept + airline_slope * response_time_range,
        'airline': airline
    })])

# Plotting the simple slope plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='response_time_hours_normalized', y='sentiment_change', hue='airline', data=plot_data)
plt.xlabel('Response Time (hours)')
plt.ylabel('Sentiment Change')
plt.title('Simple Slope Plot for Sentiment Change by Response Time and Airline')
plt.legend(title='Airline:')
plt.show()


# Setting 'KLM' as the reference category for the 'airline' variable
df['airline'] = df['airline'].astype('category')
df['airline'] = df['airline'].cat.reorder_categories(
    ['KLM'] + [cat for cat in df['airline'].cat.categories if cat != 'KLM']
)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Running the regression model with interaction term and 'KLM' as the reference category
reg2 = smf.ols(formula='sentiment_change ~ response_time_hours * C(airline)', data=train).fit()

# Printing the summary of the regression model
print(reg2.summary())

# Predicting on the test set
test['predicted_sentiment_change'] = reg2.predict(test)

# Calculating metrics
mse = mean_squared_error(test['sentiment_change'], test['predicted_sentiment_change'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(test['sentiment_change'], test['predicted_sentiment_change'])
r2 = r2_score(test['sentiment_change'], test['predicted_sentiment_change'])
adjusted_r2 = 1 - (1-r2)*(len(test)-1)/(len(test)-test.shape[1]-1)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
print(f"Adjusted R-squared: {adjusted_r2}")

# Plotting Mean Squared Error
plt.figure(figsize=(10, 6))
plt.scatter(test['response_time_hours'], test['sentiment_change'], color='blue', label='Actual')
plt.scatter(test['response_time_hours'], test['predicted_sentiment_change'], color='red', label='Predicted')
plt.xlabel('Response Time (hours)')
plt.ylabel('Sentiment Change')
plt.title('Actual vs Predicted Sentiment Change')
plt.legend()
plt.show()

# Plotting Residuals
plt.figure(figsize=(10, 6))
plt.scatter(test['predicted_sentiment_change'], test['sentiment_change'] - test['predicted_sentiment_change'], color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Sentiment Change')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()


# Filter data for specific airlines
airlines_of_interest = ['KLM', 'British_Airways', 'airfrance', 'lufthansa', 'AmericanAir', 'VirginAtlantic']
filtered_data = df[df['airline'].isin(airlines_of_interest)]

# Create a boxplot using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='airline', y='sentiment_change', data=filtered_data, palette='Set3')
plt.title('Sentiment Change Distribution by Airline', fontsize=25)
plt.xlabel('Airline', fontsize=20)
plt.ylabel('Sentiment Change', fontsize=20)
plt.xticks(fontsize=11)
plt.show()

