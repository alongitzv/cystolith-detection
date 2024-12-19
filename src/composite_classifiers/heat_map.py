import os
import numpy as np
y_test = np.repeat([0,1,2,0,1,2],[129,4,2,0,138,13])
y_pred = np.repeat([0,0,1,1,2,2],[129,4,2,0,138,13])


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
 
# Change figure size and increase dpi for better resolution
# and get reference to axes object
fig, ax = plt.subplots(figsize=(8,6), dpi=100)
 
from sklearn.metrics import confusion_matrix
 
# Order of the input parameters is important: 
# first param is the actual output values
# second param is what our model predicted
conf_matrix = confusion_matrix(y_test, y_pred)
 


# initialize using the raw 2D confusion matrix 
# and output labels (in our case, it's 0 and 1)
display = ConfusionMatrixDisplay(conf_matrix, display_labels=[0,1,2])
 
# set the plot title using the axes object
ax.set(title='Confusion Matrix for the Diabetes Detection Model')
 
# show the plot. 
# Pass the parameter ax to show customizations (ex. title) 
display.plot(ax=ax);

print(display)


import seaborn as sns
 
# Change figure size and increase dpi for better resolution
plt.figure(figsize=(8,6), dpi=100)
# Scale up the size of all text
sns.set(font_scale = 1.1)
 
# Plot Confusion Matrix using Seaborn heatmap()
# Parameters:
# first param - confusion matrix in array format   
# annot = True: show the numbers in each heatmap cell
# fmt = 'd': show numbers as integers. 
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', )
 
# set x-axis label and ticks. 
ax.set_xlabel("Predicted detection", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['fake', 'real', 'no detection'])
 
# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['fake', 'real', 'no detection'])
 
# set plot title
ax.set_title("Confusion Matrix for the Diabetes Detection Model", fontsize=14, pad=20)
 
plt.show()