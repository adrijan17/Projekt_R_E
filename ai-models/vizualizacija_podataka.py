import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from postavke import file_path, features_to_check


data = pd.read_csv(file_path)

print("="*60)
print("Vizualizacija Podataka")
print("="*60)


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(data['NDVI'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('NDVI')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('NDVI Distribution')
axes[0, 0].axvline(0.2, color='r', linestyle='--', label='Typical veg. min')
axes[0, 0].axvline(0.8, color='r', linestyle='--', label='Typical veg. max')
axes[0, 0].legend()

if 'LST_Day' in data.columns:
    axes[0, 1].scatter(data['LST_Day'], data['NDVI'], alpha=0.3, s=1)
    axes[0, 1].set_xlabel('LST_Day (Kelvin)')
    axes[0, 1].set_ylabel('NDVI')
    axes[0, 1].set_title('NDVI vs Daytime Temperature')

if 'NDWI' in data.columns:
    axes[1, 0].scatter(data['NDWI'], data['NDVI'], alpha=0.3, s=1)
    axes[1, 0].set_xlabel('NDWI')
    axes[1, 0].set_ylabel('NDVI')
    axes[1, 0].set_title('NDVI vs Water Index')

corr_matrix = data[features_to_check + ['NDVI']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Feature Correlations')

plt.tight_layout()
plt.savefig('vizualzizacija_dataseta.png', dpi=300, bbox_inches='tight')
print("Saved: vizualzizacija_dataseta.png")
plt.show()