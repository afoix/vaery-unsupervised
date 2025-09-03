#%%
import spatialdata as sd
sdata = sd.read_zarr("/mnt/efs/aimbl_2025/student_data/S-KM/wlog2/450_wlog2")
# %%
sdata["rasterized"] = sd.rasterize(
    sdata.shapes["xml_contours_450"],
    ["x", "y"],
    target_coordinate_sytem = "aligned_450",
    target_unit_to_pixels = 1
)