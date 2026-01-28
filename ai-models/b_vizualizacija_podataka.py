import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from _postavke import file_path, features_to_check, visualisaton_path


data = pd.read_csv(file_path)

print("="*60)
print("Vizualizacija Podataka")
print("="*60)


# distribucija NDVI 
print("\n1/2 - Generiranje grafa distribucije NDVI...")
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.hist(data['NDVI'], bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('NDVI')
ax1.set_ylabel('Frequency')
ax1.set_title('NDVI Distribution')
plt.tight_layout()
plt.savefig(visualisaton_path + 'ndvi_distribucija.png', dpi=300, bbox_inches='tight')
print("Saved: ndvi_distribucija.png")
plt.close()


# korelacija varijabli
print("\n2/2 - Generiranje heatmape korelacije varijabli...")
fig4, ax4 = plt.subplots(figsize=(12, 10))
corr_matrix = data[features_to_check + ['NDVI']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax4, cbar_kws={'shrink': 0.8}, 
            annot_kws={'fontsize': 20})
ax4.set_title('Feature Correlations', fontsize=16, fontweight='bold')
ax4.set_xticklabels(ax4.get_xticklabels(), fontsize=14)
ax4.set_yticklabels(ax4.get_yticklabels(), fontsize=14)
plt.tight_layout()
plt.savefig(visualisaton_path + 'korelacija_varijabli.png', dpi=300, bbox_inches='tight')
print("Saved: korelacija_varijabli.png")
plt.close()

# extra
# NDVI vs Daytime Temperature
if 'LST_Day' in data.columns:
    print("\n2/4 - Creating NDVI vs LST_Day plot...")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(data['LST_Day'], data['NDVI'], alpha=0.3, s=1)
    ax2.set_xlabel('LST_Day (Kelvin)')
    ax2.set_ylabel('NDVI')
    ax2.set_title('NDVI vs Daytime Temperature')
    plt.tight_layout()
    plt.savefig(visualisaton_path + 'vizualizacija_ndvi_vs_lst_day.png', dpi=300, bbox_inches='tight')
    print("Saved: ndvi_vs_lst_day.png")
    plt.close()

# NDVI vs Water Index
if 'NDWI' in data.columns:
    print("\n3/4 - Creating NDVI vs NDWI plot...")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(data['NDWI'], data['NDVI'], alpha=0.3, s=1)
    ax3.set_xlabel('NDWI')
    ax3.set_ylabel('NDVI')
    ax3.set_title('NDVI vs Water Index')
    plt.tight_layout()
    plt.savefig(visualisaton_path + 'vizualizacija_ndvi_vs_ndwi.png', dpi=300, bbox_inches='tight')
    print("Saved: ndvi_vs_ndwi.png")
    plt.close()

# NDVI vs Albedo
if 'Albedo_Diff' in data.columns:
    print("\n4/6 - Creating NDVI vs Albedo_Diff plot...")
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.scatter(data['Albedo_Diff'], data['NDVI'], alpha=0.3, s=1)
    ax5.set_xlabel('Albedo_Diff')
    ax5.set_ylabel('NDVI')
    ax5.set_title('NDVI vs Albedo_Diff')
    plt.tight_layout()
    plt.savefig(visualisaton_path + 'vizualizacija_ndvi_vs_albedo.png', dpi=300, bbox_inches='tight')
    print("Saved: ndvi_vs_albedo.png")
    plt.close()

# NDVI vs DEM (Elevation)
if 'DEM' in data.columns:
    print("\n5/6 - Creating NDVI vs DEM plot...")
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    ax6.scatter(data['DEM'], data['NDVI'], alpha=0.3, s=1)
    ax6.set_xlabel('DEM (Elevation)')
    ax6.set_ylabel('NDVI')
    ax6.set_title('NDVI vs Digital Elevation Model')
    plt.tight_layout()
    plt.savefig(visualisaton_path + 'vizualizacija_ndvi_vs_dem.png', dpi=300, bbox_inches='tight')
    print("Saved: ndvi_vs_dem.png")
    plt.close()

print("\n" + "="*60)
print("Vizualizacije uspješno generirane!")
print("="*60)