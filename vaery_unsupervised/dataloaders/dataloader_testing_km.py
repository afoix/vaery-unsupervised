#%%
import spatialdata as sd
sdata = sd.read_zarr("/mnt/efs/aimbl_2025/student_data/S-KM/wlog2/450_wlog2")
sdata
from spatialdata.transformations import (
    Affine,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    get_transformation,
    set_transformation,
)
# %%
poly = sdata.shapes['xml_contours_450'].iloc[0]["geometry"]
poly.centroid.coords[0]
# %%
poly.bounds[0]
#%%

sd.transformations.get_transformation(sdata.shapes["xml_contours_450"], to_coordinate_system="aligned_450", get_all=False)#

#%%
from spatialdata.dataloader import ImageTilesDataset
size=64
dataset = ImageTilesDataset(
    sdata, 
    regions_to_images = {"xml_contours_450":"czi_450"}, 
    regions_to_coordinate_systems= {"xml_contours_450": "aligned_450"}, 
    tile_scale = 1, tile_dim_in_units = size
)

dataset[12]
# %%
idx = 12
dataset[idx]
#%%
sdata["xml_contours_450"].loc["C8"]
dataset[idx]
# %%
unique_id = sdata["xml_contours_450"].iloc[idx].name
unique_id
# %%
crop_mins = dataset.tiles_coords[["miny","minx"]].iloc[idx].to_numpy()
crop_maxs = dataset.tiles_coords[["maxy","maxx"]].iloc[idx].to_numpy()

sdata_cropped = sdata.query.bounding_box(
    axes=["y", "x"],
    min_coordinate=crop_mins,
    max_coordinate=crop_maxs,
    target_coordinate_system="aligned_450",
    filter_table=False
)

sdata_cropped
#%%
from spatialdata import join_spatialelement_table

out = join_spatialelement_table(
    sdata,
    unique_id, table_name="450_table_traj_final"

)
#%%
sdata_cropped['450_table_traj_final'] = sdata_cropped['450_table_traj_final'][sdata_cropped['450_table_traj_final'].obs["lmd_well"] == unique_id]
sdata_cropped['450_table_traj_final'].obs

# %%
sdata_cropped.pl.render_images("czi_450").pl.render_shapes("xml_contours_450").pl.show()
#sdata_cropped.pl.render_shapes("xml_contours_450").pl.show()
# %%
sdata_cropped
# %%
matrix = sd.transformations.get_transformation(sdata["xml_contours_450"], to_coordinate_system="aligned_450")
matrix
# %%
sdata["affinetransformed_contours_450"] = sd.transform(
    sdata["xml_contours_450"],
    to_coordinate_system='aligned_450',
)
# %%
sdata["affinetransformed_contours"]
# %%
def load_sdata_objects(list_of_files): # make lookup table
    pass

def make_lookup(idx): # -> which dataset to look at and what idx in that dataset
    pass

def get_sample(idx): # -> image, mask, well_ID
    pass

