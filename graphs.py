import matplotlib.pyplot as plt
import numpy as np

# Δεδομένα με Mlp 1 στο τέλος
models = ['DGM', 'gen highway', 'Res net', 'Mlp 3', 'Mlp 2', 'Mlp 1']
time = np.array([5723, 3647, 2364, 2057, 1924, 1767])
mse = np.array([1.55E-06, 1.95E-06, 2.30E-06, 2.52E-06, 3.80E-06, 1.84E-03])
mae = np.array([6.40E-04, 6.672346E-04, 8.80E-04, 7.24E-04, 1.26E-03, 2.23E-02])

def plot_histogram(models, time, metric, metric_name, filename):
    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.bar(x - width/2, metric, width, label=metric_name, color=color1)
    ax1.set_ylabel(metric_name, color=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.bar(x + width/2, time, width, label='Training time', color=color2)
    ax2.set_ylabel('Training time', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f'Training time and {metric_name}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Περίπτωση 1: Συμπεριλαμβάνοντας MLP 1 (στο τέλος)
plot_histogram(models, time, mse, 'MSE', 'training_time_vs_mse_with_mlp1.png')
plot_histogram(models, time, mae, 'MAE', 'training_time_vs_mae_with_mlp1.png')

# Περίπτωση 2: Χωρίς MLP 1
mask = np.array([m != 'Mlp 1' for m in models])
models_wo_mlp1 = np.array(models)[mask]
time_wo_mlp1 = time[mask]
mse_wo_mlp1 = mse[mask]
mae_wo_mlp1 = mae[mask]

plot_histogram(models_wo_mlp1, time_wo_mlp1, mse_wo_mlp1, 'MSE', 'training_time_vs_mse_without_mlp1.png')
plot_histogram(models_wo_mlp1, time_wo_mlp1, mae_wo_mlp1, 'MAE', 'training_time_vs_mae_without_mlp1.png')

print("Plots saved successfully!")

