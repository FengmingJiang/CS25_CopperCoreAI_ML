import rasterio
import os

class GeoTIFFLoader:
    def __init__(self, paths: list):
        self.tiffs = {
            os.path.splitext(os.path.basename(p))[0]: rasterio.open(p)
            for p in paths if os.path.exists(p)
        }

    def get_tiff_by_field(self, field: str):
        return self.tiffs.get(field, None)

    def available_fields(self):
        return list(self.tiffs.keys())