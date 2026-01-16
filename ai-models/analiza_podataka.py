import pandas as pd
from postavke import file_path, features_to_check

data = pd.read_csv(file_path)

print("="*60)
print("Karakteristike Dataseta")
print("="*60)


print("\nDataset Shape:")
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")


print("\nHead:")
print(data.head())


print("\nMissing Values:")
print(data.isnull().sum())
print(f"\nUkupno: {data.isnull().sum().sum()}\n")


print("\nNDVI Statistics:")
print(data['NDVI'].describe())
# print(f"NDVI Range: [{data['NDVI'].min():.3f}, {data['NDVI'].max():.3f}]")
print("(Zdrava biljka obično između 0.2 i 0.8)")


print("\nCorrelation with NDVI:")
correlations = data[features_to_check + ['NDVI']].corr()['NDVI'].sort_values(ascending=False)
print(correlations)


