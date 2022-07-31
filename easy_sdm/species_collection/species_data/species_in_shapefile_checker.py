from pathlib import Path

import geopandas as gpd


class SpeciesInShapefileChecker:
    def __init__(self, gdf_region: Path):
        self.gdf_region = gdf_region

    def get_points_inside(self, gdf: gpd.GeoDataFrame):
        """get if points are inside dataframe
        Inspired for :https://www.matecdev.com/posts/point-in-polygon.html

        Args:
            df ([pd.DataFrame]): dataframe with lat lon values
        """
        points_in_polygon = gdf.sjoin(self.gdf_region, predicate="within", how="inner")
        new_gdf = points_in_polygon[list(gdf.columns)]
        new_gdf = new_gdf.reset_index(drop=True)
        return new_gdf
