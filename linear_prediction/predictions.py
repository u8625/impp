import pandas as pd
import matplotlib.pyplot as plt

# User Inputs
gender = input('Male or Female predictions? ')
value = int(input('Enter a weight: '))

# Importing the data
df = pd.read_csv('Machine-Learning\\old\\linear_regression\\weight_height.csv')

# Cleaning the data
df = df[df.Gender == gender]

# Defining the variables to be graphed
x = df.Weight
y = df.Height

# Generating a scatterplot
plt.scatter(x, y)

plt.show()

# Calculating SSxx
xmean = x.mean()
df['diffx'] = xmean - x
df['diffx_squared'] = df.diffx**2
SSxx = df.diffx_squared.sum()

# Calculating SSxy
ymean = y.mean()
df['diffy'] = ymean - y
SSxy = (df.diffx * df.diffy).sum()

# Calculating the slope (x)
m = SSxy/SSxx

# Calculating the intercept (b)
b = ymean - m * xmean

# Predicting values
def predict(value):
    
    prediction = m * value + b
    prediction_output = str(prediction)
    print(prediction_output[:5])
    
predict(value)
