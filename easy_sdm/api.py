from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]


if __name__ == "__main__":
    rr = RasterReader()
    rs = RasterClip()

    raster_test = rr.read(
        Path.cwd() / "data/raw/rasters/Envirem_Rasters/envir1_annual_PET.tif"
    )
    import pdb

    pdb.set_trace()
    profile = rs.clip(raster_test)
