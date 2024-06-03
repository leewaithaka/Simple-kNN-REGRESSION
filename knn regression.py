# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
%matplotlib inline
# Read the data from the file "Advertising.csv"
filename = 'Advertising.csv'
df_adv = pd.read_csv('/home/Advertising.csv')
# Take a quick look of the dataset
print(df_adv.head())
### edTest(test_findnearest) ###
# Define a function that finds the index of the nearest neighbor 
# and returns the value of the nearest neighbor.  
# Note that this is just for k = 1 where the distance function is 
# simply the absolute value.

def find_nearest(array,value):
    # Calculate absolute differences between each element in the array and the value
    differences = np.abs(array - value)
    # Find the index of the smallest difference
    idx = differences.idxmin()
    
    # Hint: To find idx, use .idxmin() function on the series
    idx = pd.Series(np.abs(array-value)).___ 

    # Return the nearest neighbor index and value
    return idx, array[idx]
# Get a subset of the data i.e. rows 5 to 13
# Use the TV column as the predictor
x_true = df_adv.TV.iloc[5:13]

# Use the Sales column as the response
y_true = df_adv.Sales.iloc[5:13]

# Sort the data to get indices ordered from lowest to highest TV values
idx = np.argsort(x_true).values 

# Get the predictor data in the order given by idx above
x_true  = x_true.iloc[idx].values

# Get the response data in the order given by idx above
y_true  = y_true.iloc[idx].values
# Display the sorted predictor and response data
print("Sorted TV values (predictor):", x_true)
print("Sorted Sales values (response):", y_true)
# Create some synthetic x-values (might not be in the actual dataset)
# Given sorted x_true and y_true from the previous step
x_true = np.array([8.6, 39.5, 43.1, 44.5, 48.3, 66.9, 149.7, 151.5, 175.1])
y_true = np.array([5.3, 10.1, 10.4, 10.4, 11.8, 10.5, 16.0, 18.5, 15.2])
x = np.linspace(np.min(x_true), np.max(x_true))

# Initialize the y-values for the length of the synthetic x-values to zero
y = np.zeros((len(x)))
# Display the synthetic x-values and initialized y-values
print("Synthetic x-values:", x)
print("Initialized y-values:", y)
# Apply the KNN algorithm to predict the y-value for the given x value
for i, xi in enumerate(x):
    nearest_idx = np.argmin(np.abs(x_true - xi))
    y[i] = y_true[nearest_idx]
# Plot the synthetic data along with the predictions    
plt.plot(x, y, '-.', label='Predicted Sales (k=1)')

# Plot the original data using black x's.
plt.plot(x_true, y_true, 'kx', label='Actual Sales')

# Set the title and axis labels
plt.title('TV vs Sales')
plt.xlabel('TV budget in $1000')
plt.ylabel('Sales in $1000')
# Read the data from the file "Advertising.csv"
data_filename = 'Advertising.csv'
df = pd.read_csv(data_filename)

# Set 'TV' as the 'predictor variable'   
x = df[['TV']]

# Set 'Sales' as the response variable 'y' 
y = df['Sales']
### edTest(test_shape) ###

# Split the dataset into training and testing with 60% training set 
# and 40% testing set with random state = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=42)

# Display the shapes of the training and testing sets to verify
print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
# Explanation:
# train_test_split(x, y, train_size=0.6, random_state=42): Splits the dataset into training and testing sets, with 60% of the data used for training and 40% for testing. The random_state=42 ensures reproducibility of the split.
# x_train, x_test, y_train, y_test: These variables store the training and testing sets for predictors (x) and response variables (y).
# print("Shape of x_train:", x_train.shape) and similar lines: These print statements display the shapes of the training and testing sets, which helps verify that the split was done correctly.
# Make sure to replace x and y with the appropriate variables containing the predictor and response data.
### edTest(test_nums) ###

# Choose the minimum k value based on the instructions given on the left
k_value_min = 1

# Choose the maximum k value based on the instructions given on the left
k_value_max = 70

# Create a list of integer k values between k_value_min and k_value_max using linspace
k_list = np.linspace(k_value_min, k_value_max, 70, dtype=int)

# Display the generated list of k values
print("List of k values:", k_list)
# Set the grid to plot the values
fig, ax = plt.subplots(figsize=(10, 6))

# Variable used to alter the linewidth of each plot
j = 0

# Loop over all the k values
for k_value in k_list:
    # Creating a kNN Regression model 
    model = KNeighborsRegressor(n_neighbors=int(k_value))
    
    # Fitting the regression model on the training data 
    model.fit(x_train, y_train)
    
    # Use the trained model to predict on the test data 
    y_pred = model.predict(x_test)
    
    # Helper code to plot the data along with the model predictions
    colors = ['grey', 'r', 'b']
    if k_value in [1, 10, 70]:
        xvals = np.linspace(x['TV'].min(), x['TV'].max(), 100).reshape(-1, 1)
        ypreds = model.predict(xvals)
        ax.plot(xvals, ypreds, '-', label=f'k = {int(k_value)}', linewidth=j+2, color=colors[j])
        j += 1

ax.legend(loc='lower right', fontsize=20)
ax.plot(x_train, y_train, 'x', label='train', color='k')
ax.set_xlabel('TV budget in $1000', fontsize=20)
ax.set_ylabel('Sales in $1000', fontsize=20)
plt.tight_layout()
plt.show()








