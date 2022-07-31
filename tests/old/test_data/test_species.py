def test_species_request(tmp_path, mock_map_shapefile_path):
    import geopandas as gpd

    from easy_sdm.raster_processing import (
        ShapefileLoader,
        Species,
        SpeciesGDFBuilder,
        SpeciesInShapefileChecker,
    )

    mays_code = 5290052
    output_path = tmp_path / "Zea_mays/Zea_mays.shp"
    mays = SpeciesGDFBuilder(
        Species(taxon_key=mays_code, name="Zea mays"),
        proposed_region=SpeciesInShapefileChecker(mock_map_shapefile_path),
    )
    mays_gdf = mays.get_species_gdf()
    mays.save_species_gdf(output_path)
    loaded_mays_gdf = ShapefileLoader(output_path).load_dataset()
    assert loaded_mays_gdf.shape == mays_gdf.shape
    assert isinstance(mays_gdf, gpd.GeoDataFrame)
