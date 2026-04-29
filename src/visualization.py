import matplotlib.pyplot as plt
import numpy as np

# Data for the models
models = ['SVM', 'CNN', 'Hybrid']
accuracy = [84.5, 90.3, 93.7]
precision = [82.1, 88.9, 92.5]
recall = [85.0, 91.0, 94.0]
f1_score = [83.5, 89.9, 93.2]

# Set the width of the bars
bar_width = 0.2
index = np.arange(len(models))

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each metric
bar1 = ax.bar(index - bar_width, accuracy, bar_width, label='Accuracy')
bar2 = ax.bar(index, precision, bar_width, label='Precision')
bar3 = ax.bar(index + bar_width, recall, bar_width, label='Recall')
bar4 = ax.bar(index + 2 * bar_width, f1_score, bar_width, label='F1-Score')

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Percentage (%)')
ax.set_title('Performance Comparison of Models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models)
ax.legend()

# Display values on top of bars
def add_values(bars):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval}%', ha='center', va='bottom')

add_values(bar1)
add_values(bar2)
add_values(bar3)
add_values(bar4)

# Show the plot
plt.tight_layout()
plt.show()
