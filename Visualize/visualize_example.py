import os
import visualize

path_to_predictions_gr = '/media/ps/SSD2T/models_20190211/grading'
path_to_predictions_ca = '/media/ps/SSD2T/models_20190211/cancer'
folders_gr = os.listdir(path_to_predictions_gr)
folders_ca = os.listdir(path_to_predictions_ca)

# Get data and average tiel predictions over ensemble.
df = visualize.read_tile_predictions(group_high=True,
                                     hyper_i=folders_gr,
                                     hyper_i_ca=folders_ca,
                                     epoch=['60'] * len(folders_gr),
                                     epoch_ca=['15'] * len(folders_ca),
                                     root=path_to_predictions_gr,
                                     root_ca=path_to_predictions_ca,
                                     dbpath='df_pred_database_tiles_test.xz',
                                     benign_to_cancer=False,
                                     scale_gradeprobs_to_ca=False)

# Initiate the Class.
conf = visualize.Confmask('13800664 1B', df)

# Get AUC of tile predicted cancer and true label.
conf.auc_tile(invert=False)

# Thumbnail of confmask
thumb = conf.thumbnail_confmask()
thumb.show()

# For the rest of teh methods we need the WSI and tissuemask.
conf.get_slide('/media/ps/WSI_images')
conf.get_tissuemask('/media/ps/WSI_images')

# Thumbnail of heatmap.
thumb = conf.thumnail_heatmap(height=800,
                              color_bg=255,
                              color_hg="red",
                              color_lg="yellow",
                              grey_tissue=True,
                              transperity_at_notile=True,
                              transperity_at_bg=True
                              )
thumb.show()

thumb = conf.thumnail_heatmap(height=6000,
                              color_bg=255,
                              color_hg="red",
                              color_lg="yellow",
                              grey_tissue=True,
                              transperity_at_notile=True,
                              transperity_at_bg=False
                              )
thumb.show()

# Thumnail of whole slide image.
thumb = conf.thumbnail_wsi(height=6000, bbox=True)
thumb.show()

# Full figure og predictions and whole slide image.
img = conf.make_img_conf_vis(height=1200, just_draw=True)
img.show()

img = conf.make_img_conf_vis(height=600, just_draw=False)
img.show()

# Make confmask suiable for web viewer.
view = conf.to_viewer('some/path')
