import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())
prod_per_year = df.groupby('year')['totalprod']
print(prod_per_year)

X = df['year']
X = X.values.reshape(-1, 1)

y = df['totalprod']
# Plot 'y' vs 'X' as a scatterplot
plt.scatter(X, y)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.title('Scatter Plot of Total Production vs Year')
plt.show()

#Create and Fit a Linear Regression Model.

#6
regr = linear_model.LinearRegression()
#7 
regr.fit(X, y)
#8
print(regr.coef_[0])
#9
y_predict = regr.predict(X)
#10
plt.plot(X, y_predict, color='red', label=' Line Regression Model')
plt.show()

#Predict the Honey Decline

#11
X_future = np.array(range(2014,  2050))
#12
X_future = X_future.reshape(-1, 1)
future_predict = regr.predict(X_future)
#13
plt.plot(X_future, future_predict, color='red', marker='o', label='Predicted Honey Production')

# Add labels and legend to the plot
plt.xlabel('Year')
plt.ylabel('Honey Production')
plt.legend()

# Show the plot
plt.show()
