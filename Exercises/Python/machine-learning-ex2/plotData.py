from scipy import linalg, sparse #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy as np #for lin alg operations
import pandas as pd #for reading in file

data = pd.read_csv('ex2data1.txt', header=None, delimiter=','); #data is a 97x2 matrix
X = data.values[:,[0,1]]
y = data.values[:, 2];

# pos = np.asmatrix(np.where(y==1))[0];
# neg = np.asmatrix(np.where(y==0))[0];


pos = np.where(y==1)[0];
neg = np.where(y==0)[0];


fig, ax = plt.subplots()

plt.plot(X[pos, 0], X[pos, 1],  'g^', label='Admitted');
plt.plot(X[neg, 0], X[neg, 1], 'bs', label='Not Admitted');

# Now add the legend with some customizations.
legend = ax.legend(loc='lower left', shadow=False);

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('1')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')


plt.xlabel("Exam 1 score");
plt.ylabel("Exam 2 score");
plt.show();
