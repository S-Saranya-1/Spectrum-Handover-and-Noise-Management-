import numpy as np
import matplotlib.pyplot as plt

# Sample data
cognitive_users = np.array([50, 100, 150, 200, 250])
throughput = np.array([2869.469, 2747.137, 2783.907, 2593.278, 2474.551])
energy_consumption = np.array([7.443, 8.506, 7.010, 8.011, 8.962])
delay = np.array([2.627, 4.673, 7.975, 10.756, 12.808])
probability_of_attack = np.array([0.861, 0.773, 0.726, 0.683, 0.581])

# Define line styles and markers
styles = {
    'throughput': {'color': '#007acc', 'marker': 'o', 'linestyle': '-'},
    'energy': {'color': '#ff5733', 'marker': 's', 'linestyle': '--'},
    'delay': {'color': '#4caf50', 'marker': 'D', 'linestyle': '-.'},
    'attack': {'color': '#9b59b6', 'marker': '^', 'linestyle': ':'},
}


# Function to create an improved plot
def improved_plot(x, y, ylabel, filename, style, ylim=None):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y,
             linestyle=style['linestyle'],
             color=style['color'],
             marker=style['marker'],
             markersize=8,
             markeredgewidth=2,
             markeredgecolor='black',
             linewidth=2,
             label='MP-FF-SH-TLC')

    plt.xlabel('Number of Cognitive Users', fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='dashed', alpha=0.6)

    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# Generate improved plots
improved_plot(cognitive_users, throughput, 'Throughput (bps)', "./Journ1/Thrghput_journ.png", styles['throughput'])
improved_plot(cognitive_users, energy_consumption, 'Energy Consumption (J)', "./Journ1/EC_journ.png", styles['energy'])
improved_plot(cognitive_users, delay, 'Delay (ms)', "./Journ1/Delay_journ.png", styles['delay'])
improved_plot(cognitive_users, probability_of_attack, 'Probability of Attack', "./Journ1/Attack_journ.png",
              styles['attack'], ylim=(0.5, 0.9))

