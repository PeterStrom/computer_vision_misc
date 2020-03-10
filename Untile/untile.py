import os
import numpy as np
import cv2
from skimage import io


class Untile:
    def __init__(self, df, scale, path_in, path_out):
        """Use predicted tiles from, say, a segmentation model and re-map them
        to their original positions in the full resolution image.

        Args:
            df: a DataFrame containing paths, image size and pixel positions.
            scale: the resize level of the tiles, 0 = no resizing, 1 = half
                the resolution etc.
            path_in: root path to prediction tiles.
            path_out: root path to store PredictionMasks.

        Return: A full resolution PredictionMask.
        """
        self.df = df
        self.scale = scale
        self.path_in = path_in
        self.path_out = path_out
        self.m = None
        self.ms = None
        self.mn = None
        self.tm = None
        self.info = df.iloc[0].copy()
        self.size = self.info.bottom - self.info.top + 1
        self.h_max = self.info.img_h
        self.w_max = self.info.img_w
        if self.scale:
            self.size = self.size // (self.scale + 1)
            self.h_max = self.info.img_h // (self.scale + 1)
            self.w_max = self.info.img_w // (self.scale + 1)

        if not os.path.exists(path_out):
            os.makedirs(path_out)

    def read_tile(self):
        pass

    def untile(self, classify=.5):
        """Read and re-map prediction tile masks.

        Args:
            classify: a threshold for binary classification. None will use
                the tiles unchanged.

        Return: a scaled PredictionMask.
        """
        if classify:
            img_type = np.uint8
        else:
            img_type = np.float

        # Generate a numpy array of full size.
        ms = np.zeros((self.h_max, self.w_max), dtype=img_type)
        mn = np.zeros((self.h_max, self.w_max), dtype=np.uint8)

        for _, i in self.df.iterrows():
            # Scanners have different pixelsize so tiling resizes X pixels
            # on the original image to the Y pixels stored on disc. This
            # means that we need to resize the predicted tile size Y back to
            # X before we re-map it to the PredictionMask.

            pred_name = i.tile_name.split('.')[0]
            pred_name = 'pred_' + pred_name + '.png'
            path = os.path.join(self.path_in, i.slide, pred_name)
            tile = cv2.imread(path, 0)
            tile = cv2.resize(tile,
                              dsize=(self.size, self.size),
                              interpolation=cv2.INTER_NEAREST)
            top = i.top
            left = i.left
            if self.scale:
                top = top // (self.scale + 1)
                left = left // (self.scale + 1)

            bottom = top + self.size
            right = left + self.size

            ms[top:bottom, left:right] += tile
            mn[top:bottom, left:right] += 1

        self.ms = ms
        self.mn = mn

    def classify_predictionmask(self, pixel_rule='any', use_probs=False):
        if pixel_rule == 'any':
            m = self.ms.copy()
            m[m > 0] = 1
            m = m.astype('uint8')
        if use_probs:
            m = self.ms.copy()
            m = np.where(self.mn, m / self.mn, m)
            m = m.astype('uint8')
        self.m = m

    def make_full_resolution(self):
        """Resizes the PredictionMask to full resolution."""
        self.m = cv2.resize(self.m,
                            (self.info.img_w, self.info.img_h),
                            cv2.INTER_NEAREST)

    def get_tissuemask(self, path_tissuemask, only_labeled_core=False):
        """Read the corresponding TissueMask.

        Args:
            path_tissuemask: (str) root path to WSI and masks.
            only_labeled_core: (bool) only labeled tissue.
        """
        inputpath = os.path.join(path_tissuemask, self.info.path)
        if only_labeled_core:
            mask_name = 'LabelMask_'
        else:
            mask_name = 'TissueMask_'

        name = mask_name + self.info.filename + '.tiff'
        path_tis = os.path.join(inputpath, name)
        self.tm = io.imread(path_tis, plugin='tifffile', as_gray=True)
        self.tm = self.tm.astype('uint8')  # 'bool' is also 8bit
        if only_labeled_core:  # LabelMask contain values > 1.
            self.tm[self.tm > 1] = 1

    def add_tissue_to_predictionmask(self):
        assert self.tm is not None, "Need to first get_tissuemask()"
        assert self.m is not None, "Need to first classify_predictionmask()"
        if not self.tm.shape == self.m.shape:
            self.tm = cv2.resize(self.tm,
                                 (self.m.shape[1], self.m.shape[0]),
                                 cv2.INTER_NEAREST)
        self.m += self.tm

    def save(self):
        name = 'PredictionMask_' + self.info.filename + '.tiff'
        path = os.path.join(self.path_out, name)
        cv2.imwrite(path, self.m)


def untile_wrap(df,
                scale,
                path_in,
                path_out,
                path_tissuemask,
                full_resolution=True):
    """Run Untile for all slides in df."""
    slides = df.slide.unique().tolist()
    for i in slides:
        ok = Untile(df[df.slide == i],
                    scale=scale,
                    path_in=path_in,
                    path_out=path_out)
        ok.untile()
        ok.get_tissuemask(path_tissuemask)
        ok.classify_predictionmask()
        ok.add_tissue_to_predictionmask()
        if full_resolution:
            ok.make_full_resolution()
        ok.save()


def untile_wrap_prob(df,
                     scale,
                     path_in,
                     path_out,
                     path_tissuemask,
                     full_resolution=False):
    """Run Untile for all slides in df."""
    slides = df.slide.unique().tolist()
    for i in slides:
        ok = Untile(df[df.slide == i],
                    scale=scale,
                    path_in=path_in,
                    path_out=path_out)
        ok.untile(classify=False)
        ok.get_tissuemask(path_tissuemask)
        ok.classify_predictionmask(use_probs=True)
        ok.add_tissue_to_predictionmask()
        if full_resolution:
            ok.make_full_resolution()
        ok.save()
