import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) SIMPLE SUCCESS-RATE BAR CHART
# ------------------------------------------------------------

methods = ['RRBT', 'Baseline']
success_rates = [9/10, 1/10]

plt.figure(figsize=(6, 4))
plt.bar(methods, success_rates)
plt.ylim(0, 1)
plt.ylabel('Success Rate')
plt.title('Task Completion Success Rate: RRBT vs Baseline')

# Add percentage labels
for i, rate in enumerate(success_rates):
    plt.text(i, rate + 0.02, f"{rate*100:.0f}%", ha='center')

plt.tight_layout()
plt.savefig("success_rate_chart.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 2) STACKED BAR CHART (SUCCESS + FAILURE MODES)
# ------------------------------------------------------------

labels = ['RRBT', 'Baseline']

# Heights of each stacked segment
success = [9, 1]
grasp_fail = [1, 4]      # grasp slip / missed grasps
bin_fail = [0, 5]        # incorrect bin predictions

x = range(len(labels))

plt.figure(figsize=(7, 6))

# Plot the stacked bars
plt.bar(x, success, color='green', label='Success')
plt.bar(x, grasp_fail, bottom=success, color='yellow', label='Grasp-related Failure')
plt.bar(x, bin_fail,
        bottom=[success[i] + grasp_fail[i] for i in x],
        color='red', label='Incorrect Bin Prediction')

# Labels & axes
plt.xticks(x, labels, fontsize=12)
plt.yticks(range(0, 11), fontsize=12)
plt.ylabel('Number of Trials (out of 10)', fontsize=12)
plt.title('Task Outcomes for RRBT vs Baseline', fontsize=14)

# Legend
plt.legend(title='Outcome Categories')

plt.tight_layout()
plt.savefig("stacked_outcomes_chart.png", dpi=300)
plt.show()
