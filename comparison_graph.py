import numpy as np
import matplotlib.pyplot as plt

# Datasets
datasets = ["D1-Swiss", "D2-Cultural", "D3-Academic", "D4-Tech"]

# Silhouette scores (AE, PCA) - raw values
ae_sil = np.array([0.3900974690914154,
    0.5764933824539185,
    0.38219380378723145,
    0.47473376989364624])
pca_sil = np.array([0.5727439130110569,
    0.8656218913689221,
    0.6601344973576091,
    0.5142611372741102])


# Raw reconstruction MSEs (AE test MSEs and PCA test MSEs)
ae_mse = np.array([
    1.0954,  # D1-Swiss
    0.8954,  # D2-Cultural
    0.9991,  # D3-Academic
    0.8371   # D4-Tech
]) 
pca_mse = np.array([
    0.383119,  # D1-Swiss
    0.641311,  # D2-Cultural
    0.575375,  # D3-Academic
    0.601701   # D4-Tech
])

ind = np.arange(len(datasets))
ind = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))

# Four bars per dataset: AE_sil, PCA_sil, AE_mse, PCA_mse
group_width = 0.8
bar_width = group_width / 4.0

# Offsets for the 4 bars within each group
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

bars = []
bars.append(ax.bar(ind + offsets[0], ae_sil, bar_width, label='AE (silhouette)', color='#FFB3B3', edgecolor='k'))
bars.append(ax.bar(ind + offsets[1], pca_sil, bar_width, label='PCA (silhouette)', color='#B3D9FF', edgecolor='k'))
bars.append(ax.bar(ind + offsets[2], ae_mse, bar_width, label='AE (recon MSE)', color='#FF7F6F', edgecolor='k'))
bars.append(ax.bar(ind + offsets[3], pca_mse, bar_width, label='PCA (recon MSE)', color='#4DA6FF', edgecolor='k'))

ax.set_xticks(ind)
ax.set_xticklabels(datasets, rotation=20)
ax.set_title('AE vs PCA — Silhouette and Reconstruction MSE (all bars)')

# Since silhouette and MSE are on different scales, show a secondary y-axis with ticks for silhouette
ax.set_ylabel('Value (silhouette scores and MSE)')

# Annotate all bars
for group in bars:
	for b in group:
		h = b.get_height()
		ax.text(b.get_x() + b.get_width()/2, h + 0.01 * max(ae_mse.max(), pca_mse.max(), 1.0), f'{h:.4f}', ha='center', va='bottom', fontsize=8)

ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
out_path = 'pca_ae_all_bars.png'
plt.savefig(out_path, dpi=200)
print(out_path)
plt.show()
