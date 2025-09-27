import matplotlib.pyplot as plt
import numpy as np

# Data for the models
models = ['Naïve Bayes', 'Hybrid NB + MLP']
f1_scores = [0.9793, 0.9993]
# accuracy_scores = [0.9785, 0.9993] # You can use this if you prefer to plot accuracy

# Create the bar chart for F1-scores
plt.figure(figsize=(8, 6)) # Adjust figure size as needed
bars = plt.bar(models, f1_scores, color=['skyblue', 'lightcoral'])

# Add labels and title
plt.xlabel('Spam Detection Models', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('Figure 4.3: Comparison of F1-Scores for Spam Detection Models', fontsize=14)
plt.ylim(0.95, 1.005) # Adjust y-axis limits to better highlight differences if scores are close
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add the F1-score values on top of each bar for clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.savefig('Figure_4_3_F1_Score_Comparison.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# --- Optional: Code to plot Accuracy scores instead ---
# If you want to plot accuracy scores, you can comment out the F1-score plotting
# block above and uncomment the block below (or run it separately).

# plt.figure(figsize=(8, 6))
# bars_accuracy = plt.bar(models, accuracy_scores, color=['skyblue', 'lightcoral'])
# plt.xlabel('Spam Detection Models', fontsize=12)
# plt.ylabel('Accuracy Score', fontsize=12)
# plt.title('Figure 4.3: Comparison of Accuracy Scores for Spam Detection Models', fontsize=14)
# plt.ylim(0.95, 1.005) # Adjust y-axis limits
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# for bar in bars_accuracy:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig('Figure_4_3_Accuracy_Comparison.png', dpi=300, bbox_inches='tight')
# plt.show()