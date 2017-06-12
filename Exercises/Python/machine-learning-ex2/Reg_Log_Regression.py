import scipy.optimize as opt 
from scipy import linalg, sparse, optimize #for linalg
import matplotlib.pyplot as plt #for the graph
import numpy #for lin alg operations
import pandas as pd #for reading in file

data = pd.read_csv('ex2data1.txt', header=None, delimiter=',')

pos = data[data[2].isin([1])] #creates matrix with only admitted scores
neg = data[data[2].isin([0])]

fig, ax = plt.subplots(figsize=(8,8))  


plt.title("Admittance Based On Exam Scores")
ax.scatter(pos[0], pos[1], s=50, c='g', marker='o', label='Accepted')  
ax.scatter(neg[0], neg[1], s=50, c='r', marker='x', label='Rejected')  
ax.legend()
ax.set_xlabel('Exam 1 Score');  
ax.set_ylabel('Exam 2 Score') 
plt.show();

