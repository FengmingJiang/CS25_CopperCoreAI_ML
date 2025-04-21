import pandas as pd

class CSVInput:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def has_model_features(self, required_features: list) -> bool:
        return set(required_features).issubset(set(self.df.columns))

    def has_coordinates(self) -> bool:
        return "latitude" in self.df.columns and "longitude" in self.df.columns

    def get_df(self) -> pd.DataFrame:
        return self.df