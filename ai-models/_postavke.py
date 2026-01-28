file_path = 'data/dataset_v3.csv'
# NASA MODIS remote sensing products and SRTM DEM data for the Marmara Region of Turkey.

# @misc{umut_alkan_2025,
#  title        = {MODIS LST \& DEM (Marmara)},
#  url          = {https://www.kaggle.com/dsv/11200689},
#  DOI          = {10.34740/KAGGLE/DSV/11200689},
#  publisher    = {Kaggle},
#  author       = {Umut Alkan},
#  year         = {2025}
# }


features_to_check = ['LST_Day', 'LST_Night', 'LST_Diff', 'NDWI', 'Albedo_Diff', 'DEM', 'Sun_Angle', 'EVI']
# izbacujem LST_Night zbog redundancije s LST_Day, LST_Diff i EVI jer je taj podatak vrlo sličan NDVI 
feature_columns = ['LST_Day', 'LST_Diff', 'NDWI', 'Albedo_Diff', 'DEM', 'Sun_Angle']

visualisaton_path = "vizualizacija/"
output_data_path = "metrike modela/"