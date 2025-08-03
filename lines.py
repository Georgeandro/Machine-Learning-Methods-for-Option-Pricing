import numpy as np
import matplotlib.pyplot as plt

# Original data
models = ['DGM', 'gen highway', 'Res net', 'Mlp 3', 'Mlp 2', 'Mlp 1']
time = np.array([5723, 3647, 2364, 2057, 1924, 1767])
mse = np.array([1.55E-06, 1.95E-06, 2.30E-06, 2.52E-06, 3.80E-06, 1.84E-03])
mae = np.array([6.40E-04, 6.672346E-04, 8.80E-04, 7.24E-04, 1.26E-03, 2.23E-02])

# Remove MLP 1
mask = np.array([m != 'Mlp 1' for m in models])
models = np.array(models)[mask]
time = time[mask]
mse = mse[mask]
mae = mae[mask]

# Normalize
time_norm = time / np.max(time)
mse_norm = mse / np.max(mse)
mae_norm = mae / np.max(mae)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(time_norm, mse_norm, color='blue', label='MSE')
plt.scatter(time_norm, mae_norm, color='green', label='MAE')
plt.plot(time_norm, mse_norm, color='blue', linestyle='--')
plt.plot(time_norm, mae_norm, color='green', linestyle='--')
plt.xlabel('Normalized Training time')
plt.ylabel('Normalized error')
plt.title('Normalized MSE and MAE vs Training time (no MLP 1)')
plt.legend()
plt.tight_layout()
plt.savefig('normalized_mse_mae_vs_time_no_mlp1.png')
plt.close()

# Print normalized points
print("\nNormalized MSE vs training time points (no MLP 1):")
for m, t, e in zip(models, time_norm, mse_norm):
    print(f"{m}: time={t:.3f}, MSE={e:.3f}")

print("\nNormalized MAE vs training time points (no MLP 1):")
for m, t, e in zip(models, time_norm, mae_norm):
    print(f"{m}: time={t:.3f}, MAE={e:.3f}")

print("Plot saved as 'normalized_mse_mae_vs_time_no_mlp1.png'.")

