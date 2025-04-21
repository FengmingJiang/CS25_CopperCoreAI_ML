from scripts.io.csv_reader import CSVInput
from scripts.io.geotiff_loader import GeoTIFFLoader

class InputResolver:
    def __init__(self, csv_path, geotiff_paths, fallback_paths, feature_columns):
        self.csv_input = CSVInput(csv_path)
        self.user_tiffs = GeoTIFFLoader(geotiff_paths)
        self.fallback_tiffs = GeoTIFFLoader(fallback_paths)
        self.features = feature_columns

    def is_direct_prediction(self):
        return self.csv_input.has_model_features(self.features)

    def get_coords(self):
        if self.csv_input.has_coordinates():
            return self.csv_input.get_df()[["latitude", "longitude"]]
        raise ValueError("No coordinates found in CSV")

    def get_combined_tiffs(self):
        full_map = {}
        for feat in self.features:
            ds = self.user_tiffs.get_tiff_by_field(feat) or self.fallback_tiffs.get_tiff_by_field(feat)
            if ds:
                full_map[feat] = ds
            else:
                raise ValueError(f"Missing required geotiff: {feat}")
        return full_map