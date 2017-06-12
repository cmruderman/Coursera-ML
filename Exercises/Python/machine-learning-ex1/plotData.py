from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy #for lin alg operations
import pandas as pd #for reading in file

data = pd.read_csv('ex1data1.txt', header=None, delimiter=',');
X=data.values[:,0]; #zero based columns
y=data.values[:,1];

plt.title("Plotting");
plt.ylabel("Profit in $10,000s");
plt.xlabel("Population of City in 10000s");
plt.plot(X, y, 'ro');
plt.axis([4, 25, -5, 25]);
plt.show();
