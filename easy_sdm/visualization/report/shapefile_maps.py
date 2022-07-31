import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


class ShapefilePlotter:
    def __init__(self) -> None:
        pass

    def plot_points_on_country(
        self,
        species_name: str,
        map_result_path: str,
        species_presences_path: str,
        species_absences_path: str = None,
    ):

        fig, ax = plt.subplots(figsize=(10, 10))

        self._brazil_country_level_gpd.plot(ax=ax, facecolor="gray")

        if ".csv" in species_presences_path:
            df_species_presences = pd.read_csv(species_presences_path)
            gdf_species_presences = gpd.GeoDataFrame(
                df_species_presences,
                geometry=gpd.points_from_xy(
                    df_species_presences.LONGITUDE, df_species_presences.LATITUDE
                ),
                crs="epsg:4326",
            )
            # df_species_presences['geometry'] = df_species_presences['geometry'].apply(wkt.loads)
            # gdf_species_presences = gpd.GeoDataFrame(df_species_presences, crs='epsg:4326')
        elif ".shp" in species_presences_path:
            gdf_species_presences = gpd.read_file(species_presences_path)
            gdf_species_presences.plot(
                ax=ax, color="blue", markersize=5, label="presences"
            )

        if species_absences_path:
            if ".csv" in species_absences_path:
                df_species_absences = pd.read_csv(species_absences_path)
                gdf_species_absences = gpd.GeoDataFrame(
                    df_species_absences,
                    geometry=gpd.points_from_xy(
                        df_species_absences.LONGITUDE, df_species_absences.LATITUDE
                    ),
                    crs="epsg:4326",
                )
            elif ".shp" in species_absences_path:
                gdf_species_absences = gpd.read_file(species_absences_path)
                gdf_species_absences.plot(
                    ax=ax, color="red", markersize=5, label="absences"
                )
                plt.title(
                    f"Ocorrências e Abeências da espécie \n {species_name} no Brasil",
                    fontsize=20,
                )
            else:
                plt.title(
                    f"Ocorrências da espécie  \n {species_name} no Brasil", fontsize=20
                )

        plt.ylabel("Latitude [graus]", fontsize=16)
        plt.xlabel("Longitude [graus]", fontsize=16)
        plt.tight_layout()
        plt.savefig(map_result_path)
