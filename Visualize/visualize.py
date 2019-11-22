import os
import cv2
import openslide
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image, ImageOps
import trainslide as ts
from sklearn import metrics


def viz_tiles(path_tiles, df_val):
    """Kimmo fill in"""
    df_val = df_val[df_val.GS_tile == '5 + 5']
    df_val = df_val[df_val.label.str.startswith('Cancer')]
    df_val.reset_index(inplace=True)
    for i in range(len(df_val)):
        filename = os.path.join(path_tiles, df_val['slide'][i], df_val['tile_name'][i])
        filename = filename.replace(".png", ".jpg")
        img = io.imread(filename)

        plt.imshow(img)
        plt.title(df_val['tile_name'][i] + '\n True y: ' + str(int(df_val['y'][i])))
        plt.xlabel("Estimated probability by class: \n" +
                   str(df_val['tile_pred_class_0'][i]) + ' - ' +
                   str(df_val['tile_pred_class_1'][i]) + ' - ' +
                   str(df_val['tile_pred_class_2'][i]) + ' - ')
        # str(df_val['tile_pred_class_3'][i]) + ' - ')
        plt.draw()
        plt.waitforbuttonpress()
    return None


def read_tile_predictions(group_high,
                          hyper_i,
                          epoch,
                          root,
                          hyper_i_ca=None,
                          epoch_ca=None,
                          root_ca=None,
                          dbpath=None,
                          benign_to_cancer=True,
                          scale_gradeprobs_to_ca=True):
    """
    Read an ensemble of dataframes with tile predictions. If also reading only
    cancer predictions then the benign predictions will be substituted.

    :param group_high: (bool) should grade 4 and 5 be combined as high grade.
    :param hyper_i: folders with (grade) predictions.
    :param epoch: witch epoch to be used, if None then the last.
    :param root: path to folders.
    :param hyper_i_ca: same as above but for cancer only.
    :param epoch_ca: same as above but for cancer only.
    :param root_ca: same as above but for cancer only.
    :param dbpath: if specific name on df (other than df_pred.xz).
    :param benign_to_cancer: should tile_pred_class_0 be pr(cancer) instead
        of pr(benign). NOTE: once the confmask is made, it is not ok to take
        the complement of the first channel. In that case you need to add the
        two other color channels to get the "complement".
    :param scale_gradeprobs_to_ca: scale probabilities according to:
        tile_pred_class_0 == tile_pred_class_1 + ... + tile_pred_class_N
        NOTE: If set to True, benign_to_cancer should also be set to True.
    :return: an averaged data frame.
    """
    warn = 'If scale_gradeprobs_to_ca=True, benign_to_cancer must be True!'
    assert not(scale_gradeprobs_to_ca and not benign_to_cancer), warn
    
    viz = ts.ensemble_parms(hyper_i=hyper_i,
                            hyper_i_ca=hyper_i_ca,
                            epoch=epoch,
                            epoch_ca=epoch_ca,
                            root=root,
                            root_ca=root_ca,
                            dbpath=dbpath)

    viz_gr = ts.ensemble_paths(viz, ca=False)
    df_gr = ts.agg_tile_prob(viz_gr)

    if viz.hyper_i_ca:
        viz_ca = ts.ensemble_paths(viz, ca=True)
        df_ca = ts.agg_tile_prob(viz_ca)
        df_gr = df_gr.drop('tile_pred_class_0', axis=1)
        df_ca = df_ca[['tile_name', 'tile_pred_class_0']]
        df_gr = pd.merge(df_gr, df_ca, how='inner', on='tile_name')

    if benign_to_cancer:
        df_gr['tile_pred_class_0'] = 1 - df_gr['tile_pred_class_0']

    if group_high:
        df_gr = ts.combine_4_5(df_gr)
    
    if scale_gradeprobs_to_ca:
        # Name of columns to scle
        cols = [name for name in df_gr.columns if name.startswith('tile_pred_class')]
        cols.sort()

        # Scale each grade's probability such that the sum of all grade 
        # probabilities equals the probability of any cancer.
        # Avoid division by zero in cases where the sum is zero.
        scaling = pd.Series(np.zeros(df_gr.shape[0],'float32'))
        sumgrades = df_gr[cols[1:]].sum(axis=1)
        scaling = scaling.where(sumgrades == 0, df_gr[cols[0]] / sumgrades)
        # Apply scaling to each column of grade probabilities.
        for coli in range(1,len(cols)):
            df_gr[cols[coli]] = scaling*df_gr[cols[coli]]

    return df_gr


def fromarray_large(array, mode=None):
    """ This replaces PIL's Image.fromarray() method for images with more than 2^28 pixels.
    Image is split into N pieces horizontally and the pieces are then pasted ( Image.paste() )
    into PIL image to circumvent using fromarray() for large images.
    Args:
        array: input image as numpy ndarray
        mode: PIL image mode for the output image, if none mode is deducted automatically
    Returns:
        Pil image that has the same size and content as the input array
    
    Original code written by Masi Valkonen (Tampere University), modified by
    Kimmo Kartasalo (Tampere University).
    """

    if mode is None:
        # get mode by creating one pixel image
        mode = Image.fromarray(array[0:1, 0:1]).mode

    # divide image into N equal sized pieces.
    # Here the N is selected to be as large as possible so that size of every piece is below 2^28.
    N = int(np.ceil((array.shape[0] * array.shape[1]) / (2 ** 28)))

    # divide array's x-axis into N slices
    piece_inds = np.linspace(0, array.shape[1], N + 1).astype(int)

    # create empty image having the same size as input array
    # PIL uses coordinates in order x, y whereas numpy uses y, x
    pil_im = Image.new(mode, array.shape[:2][::-1])  # 8-bits / channel (3x8)

    # iterate over all pieces and paste them into the output image
    y_min = 0
    y_max = array.shape[0]
    for i in range(N):
        x_min = piece_inds[i]
        x_max = piece_inds[i + 1]
        # left, upper, right, lower coordinates of the pasted image in the output coordinates
        box = (x_min, y_min, x_max, y_max)
        piece = Image.fromarray(array[:, x_min:x_max])
        pil_im.paste(piece, box)

    return pil_im


class Confmask:

    def __init__(self, slide, df, classify=False):
        """Take a list of predictions and return a matrix of predictions
        of size self.size.

        Args:
            df: A Pandas DataFrame object with Predictions for each tile
                together with tile name and bottom and right pixel.
            classify: (bool) return 1 color channel with predicted classes.

        Return: A heat map (numpy array) of same size as the image/stride with
            pixel probailites as entries.
        """
        df = df[df.slide == slide].reset_index(drop=True)

        self.slide = slide
        self.df = df
        self.path_img = df.path[0]
        self.filename = df.filename[0]
        self.ext = df.ext[0]
        self.w_max = int(df.img_w[0])
        self.h_max = int(df.img_h[0])
        self.tilesize = df.bottom[0] - df.top[0] + 1
        # TODO: Get stride from df instead, it's safer!
        self.stride = min(int(i) for i in df.top.diff() if i > 0)
        pred_cols = [name for name in df.columns if name.startswith('tile_pr')]
        pred_cols.sort()
        self.pred_names = pred_cols

        # Retrieve size of ConfMask at native low-res (downsampling by stride).
        self.CM_w, self.CM_h = (self.w_max // self.stride,
                                self.h_max // self.stride)

        # Transform full-res tile coordinates to low-res ConfMask coordinates.
        pos_img = ['top', 'bottom', 'right', 'left']
        pos_con = ['CM_top', 'CM_bottom', 'CM_right', 'CM_left']
        self.df[pos_con] = self.df[pos_img] // self.stride

        # Depth of m (number of predicted classes).
        d = len(self.pred_names)

        # Initialize a low-res ConfMask array of zeros.
        m = np.zeros((self.CM_h, self.CM_w, d), dtype=np.float16)

        # Initialize an array for keeping track of how many tiles contribute to
        # each location in the ConfMask.
        c = np.zeros((self.CM_h, self.CM_w), dtype=np.uint8)

        # For each tile (row in df), add the probabilities in the corresponding
        # locations in the low-res ConfMask. Some locations will have 
        # probabilities added several times due to overlapping tiles.
        for _, i in df.iterrows():  # tuple index, row.
            # Get the low-res coordinates where this tile maps to.
            h, w = (slice(i.CM_top, i.CM_bottom),
                    slice(i.CM_left, i.CM_right))
            
            # Add the probabilities of this tile to these locations.
            for j, k in enumerate(self.pred_names):
                m[h, w, j] += i[k]
            
            # Keep track of how many tiles contribute to this locations. 
            # Avoid overflow by converting to uint16 if needed.
            c[h, w] += 1
            if np.any(c == 255):
                c = c.astype(np.uint16)

        # Divide probabilities by the number of contributing tiles to get mean.
        # Apply to all channels (classes) and avoid division by zero.
        c = c.reshape(self.CM_h, self.CM_w, 1)
        c = c.repeat(d, axis=2)
        m[c > 0] = m[c > 0] / c[c > 0]
        m[c == 0] = np.nan

        if classify:
            m = m.argmax(axis=2)  # set both bg and benign to 0.

        self.CM = m

        # Optional attributes by set by methods.
        self.TM = None
        self.WS = None
        self.bbox = None

    @classmethod
    def from_path(cls, slide, path, combine_4_5, classify=False):
        """Read all (even if just one) dataframe with tile predictions.

        Args:
            slide: (str) the slide to make a confmask of.
            path: path to dataframe(s), reads all in the folder.
            combine_4_5: should grade 4 and 5 be combined.
            classify: (bool) return 1 color channel with predicted classes.

        Return:
            A nested dictionary: outer ensemble and inner X, and a dict of Y.

        Note: This is slow because the whole df(s) are read from disk, if many
            instances of the class are expected then first read the df(s) and
            intitiate the class by defalult method.
        """
        files = os.listdir(path)
        n = len(files)

        if n == 1:
            print('Only one file in folder, no averaging!')

        df = ts.agg_tile_prob(path=path)
        if combine_4_5:
            df = ts.combine_4_5(df)

        return cls(slide=slide, df=df, classify=classify)

    def __repr__(self):
        return 'Confmask({})'.format(self.slide)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df[item]

    def to_viewer(self, path_save, resample=Image.BILINEAR):
        assert self.TM is not None, 'first run method get_tissuemask()'

        Image.MAX_IMAGE_PIXELS = 4255906369000

        # Get the ConfidenceMask from the object.
        cm = self.CM.copy()

        # Find pixels with low, but predicted probabilities. Replace the low 
        # values with a value corresponding to 2 on uint8 scale.
        cm[cm < 2 / 255] = 2 / 255

        # Find pixels with missing probabilities, including background.
        # Assign them a value corresponding to 2 on the uint8 scale.
        cm[np.isnan(cm)] = 2 / 255

        # Convert the mask to uint8.
        cm = (cm * 255).astype(np.uint8)

        # Convert ConfMask to PIL Image and upscale to full-res by stride.
        # Size needs to be given as width, height.
        fullsize = (self.stride * cm.shape[1], self.stride * cm.shape[0])
        cm = Image.fromarray(cm).resize(fullsize, resample=resample)
        
        # Figure out how much padding is needed to compensate for the margin at
        # the bottom-right of the WSI not covered by tiles. Zero-pad.
        padding = (0, 0, self.w_max-fullsize[0], self.h_max-fullsize[1])
        cm = ImageOps.expand(cm, border=padding, fill=0)

        # Pick TissueMask from the object and invert it to get background.
        tis = self.TM == 0
        tis = 255 * np.uint8(tis)
        # Convert TissueMask to PIL Image using a workaround function for 
        # arrays larger than supported by PIL.Image.fromarray.
        if tis.shape[0] * tis.shape[1] < 2 ** 28:
            tis = Image.fromarray(tis).convert("1")
        else:
            tis = fromarray_large(tis).convert("1")

        # Set background pixels to 0.
        cm.paste((0, 0, 0), mask=tis)
        del tis

        # Write to disk.
        name = 'ConfidenceMask_' + self.slide + '.tiff'
        save_name = os.path.join(path_save, name)
        cm.save(save_name, compression="tiff_lzw")

    def get_slide(self, path_input):
        """Read and set the whole slide image to 'self'.

        Args:
            path_input: (str) root path to WSI and masks.

        Return:
            None
        """
        # Read whole slide
        inputpath = os.path.join(path_input, self.path_img)
        path = os.path.join(inputpath, self.filename + self.ext)
        self.WS = openslide.OpenSlide(path)
        
    def get_tissuemask(self, path_input):
        """Read and set the TissueMask to 'self'.

        Args:
            path_input: (str) root path to WSI and masks.

        Return:
            None
        """
        # Read tissue mask
        inputpath = os.path.join(path_input, self.path_img)
        name = 'TissueMask_' + self.filename + '.tiff'
        path_tis = os.path.join(inputpath, name)
        self.TM = io.imread(path_tis, plugin='tifffile', as_gray=True)

    def get_bbox(self,
                 half=True,
                 small=False,
                 pad=False):

        df = self.df.copy()

        if half:
            df = df[df.right > (self.w_max // 2)]

        if pad:
            pad = self.tilesize // 4

        bbox_w_min = df.left.min()
        bbox_w_max = df.right.max()
        bbox_h_min = df.top.min()
        bbox_h_max = df.bottom.max()

        bbox = [bbox_w_min - pad,
                bbox_h_min - pad,
                bbox_w_max + pad,
                bbox_h_max + pad]

        if small:
            bbox = [x // self.stride for x in bbox]

        self.bbox = bbox

    def make_img_conf_vis(self,
                          height,
                          just_draw):
        """Produce an image with tissue and various confmasks side-by-side.

        Each of the three confmask colorchannels will be displayed side-by-side,
        possibly by some transformation of the confmask array.

        Args:
            height: (int) the height in pixels of the output image
            just_draw: (bool) should regions be drawn on top of image instead of
                confmasks side-by-side

        Return: A PIL image.
        """
        assert self.WS is not None, 'No WSI in attribute, run get_slide()'
        assert self.TM is not None, 'No TissueMask in attribute, run get_tissuemask()'

        # (Re)set bbox
        self.get_bbox(half=True, small=False, pad=True)

        cm = self.CM.copy()

        cm = (cm * 255).astype(np.uint8)

        h = self.bbox[3] - self.bbox[1]
        w = self.bbox[2] - self.bbox[0]

        if w > h:
            height = height // 4

        img = self.WS.read_region((self.bbox[0], self.bbox[1]),
                                  0,
                                  size=(w, h))
        width = int(np.floor(w / h * height))  # width in output

        img = img.resize((width, height), Image.ANTIALIAS)  # resize
        img = np.array(img)[:, :, :3]  # remove alpha channel

        hw = self.TM.shape
        tis = self.TM[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]

        # Crop and set background in confmask
        b = Image.fromarray(cm).resize((hw[1], hw[0]))  # Upscale
        b = np.array(b)
        b = b[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
        b[~tis] = 255

        # Resize to output size
        b = Image.fromarray(b).resize((width, height))  # Downscale
        b = np.array(b)

        if just_draw:
            # find all the '3' and '4/5' patterns in the image
            lower3 = np.array([128, 128, 0])
            upper3 = np.array([255, 255, 128])
            lower4 = np.array([128, 0, 128])
            upper4 = np.array([255, 128, 255])

            grade3 = cv2.inRange(b, lower3, upper3)
            grade4 = cv2.inRange(b, lower4, upper4)

            _, contours3, _ = cv2.findContours(grade3,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
            _, contours4, _ = cv2.findContours(grade4,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

            out = img.copy()  # a copy is needed for some reason..
            # loop over the contours
            for c in contours3:
                cv2.drawContours(out, [c], -1, (0, 255, 0), 4)
            for c in contours4:
                cv2.drawContours(out, [c], -1, (255, 0, 0), 4)

        else:
            # Fill each of color channels in confmask to RGB
            fill = np.zeros((h, w), dtype='uint8')
            fill[~tis] = 255
            fill = Image.fromarray(fill).resize((width, height))
            fill = np.array(fill)

            # Stack together color channels side-by-side
            b1 = np.dstack((b[..., 0], b[..., 0], fill))
            b2 = np.dstack((fill, b[..., 1], fill))
            b3 = np.dstack((b[..., 2], fill, fill))

            out = np.concatenate((img, b1, b2, b3), axis=1)

        out = Image.fromarray(out)
        return out

    def auc_tile(self, invert=True):
        unknown = self.df.label.isin(['Unknown'])
        benign = self.pred_names[0]
        pred_cx = self.df[benign][~unknown]
        if invert:
            pred_cx = 1 - pred_cx
        true_cx = np.where(self.df.label == 'Cancer', 1, 0)[~unknown]

        return metrics.roc_auc_score(true_cx, pred_cx)

    def thumbnail_confmask(self, scale=None):
        assert self.CM.shape[2] in [1, 3, 4], "No understandable dimention"

        cm = self.CM.copy()

        if self.CM.shape[2] == 1:
            pass  # Need to deside how to view classified pixels.

        else:
            if self.CM.shape[2] == 4:
                msg = ["Too many predicted classes!\n",
                       self.pred_names[2],
                       "and",
                       self.pred_names[3],
                       "was combined for visualization."]
                print(" ".join(msg))
                cm_1 = cm[:, :, :2]
                cm_2 = cm[:, :, 2] + cm[:, :, 3]
                cm = np.dstack((cm_1, cm_2[..., np.newaxis]))

            cm = np.floor(cm * 255).astype('uint8')
            cm = cv2.cvtColor(cm, cv2.COLOR_RGB2BGR)

        cm = Image.fromarray(cm)

        if scale:
            newsize = tuple(int(np.floor(i * scale)) for i in cm.size)
            cm = cm.resize(newsize, Image.ANTIALIAS)

        return cm

    def thumbnail_wsi(self, height, bbox=False, alpha=False):
        """Read (and possible rescale) the whole slide image.

        Args:
            height: pixel height after rescaling.
            bbox: (bool) Crop the image if True. 'bbox' must be set before.
            alpha: (bool) if False remove the alpha channel.

        Return:
            A PIL Image.
        """
        assert self.WS is not None, 'No WSI in attribute, run get_slide()'

        if bbox:
            h = self.bbox[3] - self.bbox[1]
            w = self.bbox[2] - self.bbox[0]

            if w > h:
                height = height // 4

            img = self.WS.read_region((self.bbox[0], self.bbox[1]), 0, size=(w, h))
            width = int(np.floor(w / h * height))  # width in output
            img = img.resize((width, height), Image.ANTIALIAS)  # resize
        else:
            w, h = self.WS.dimensions
            width = int(np.floor(w / h * height))  # width in output
            img = self.WS.get_thumbnail((width, height * w / h))

        if not alpha:
            img = np.array(img)[:, :, :3]  # remove alpha channel
            img = Image.fromarray(img)

        return img

    def thumnail_heatmap(self,
                         height,
                         color_bg=255,
                         color_hg="yellow",
                         color_lg="blue",
                         grey_tissue=False,
                         transperity_at_notile=True,
                         transperity_at_bg=True,
                         classify=None):
        """Heatmap of high and low predicted grade on top of tissue.

        Args:
            height: (int) the height in pixels of the output image
            color_bg: (int: 0-255) or list of pixel values e.g. [0, 0, 255]
            color_hg: "red" or "yellow", for high grade, low grade will be
                green or blue depending on this choice.
            color_lg: ...
            grey_tissue: (bool) should H&E be substituted with greyscale.
            transperity_at_notile: (bool) should tissue with no tiles be
                transparent.
            transperity_at_bg: (bool) should background be transparant.
            classify: If None color intensity represent probability, else
                a value 0 to 1 for positivity (same color intensity in all
                pixels).

        Return: A PIL image.
        """
        assert self.WS is not None, 'No WSI in attribute, run get_slide()'
        assert self.TM is not None, 'No TissueMask in attribute, run get_tissuemask()'

        # (Re)set bbox
        self.get_bbox(half=True, small=False, pad=True)

        hw = self.TM.shape
        tis = self.TM[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]

        cm = self.CM.copy()

        # Find pixels with low, but predicted probabilities. Replace the low
        # values with a value corresponding to 2 on uint8 scale.
        cm[cm < 2 / 255] = 2 / 255

        # Find pixels with missing probabilities, including background.
        # Assign them a value corresponding to 2 on the uint8 scale.
        cm[np.isnan(cm)] = 2 / 255

        # Convert the mask to uint8.
        cm = (cm * 255).astype(np.uint8)

        h = self.bbox[3] - self.bbox[1]
        w = self.bbox[2] - self.bbox[0]

        if w > h:
            height = height // 4

        img = self.WS.read_region((self.bbox[0], self.bbox[1]),
                                  0,
                                  size=(w, h))
        width = int(np.floor(w / h * height))  # width in output

        img = img.resize((width, height), Image.ANTIALIAS)  # resize

        img = np.array(img)[:, :, :3]  # remove alpha channel
        img = Image.fromarray(img)

        # Crop and set background in confmask
        b = Image.fromarray(cm).resize((hw[1], hw[0]),
                                       Image.BILINEAR)
        b = np.array(b)
        b = b[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]

        mask = np.all(b == [2, 2, 2], axis=2) * tis

        if not classify:
            transperity = Image.fromarray(np.divide(255 - b[:, :, 0], 2).astype(np.uint8))
            tmp1 = np.where(b[:, :, 1] > b[:, :, 2], b[:, :, 1], 0)
            tmp2 = np.where(b[:, :, 2] >= b[:, :, 1], b[:, :, 2], 0)
            b[:, :, 1] = tmp1
            b[:, :, 2] = tmp2
        else:
            tres = round(classify * 255)
            transperity = Image.fromarray(np.where(b[:, :, 0] > tres, 0, 128).astype(np.uint8))
            b[:, :, 1] = np.where(b[:, :, 1] > b[:, :, 2], 255, 0)
            b[:, :, 2] = np.where(b[:, :, 2] >= b[:, :, 1], 255, 0)

        if transperity_at_notile:
            transperity = np.where(mask, 0, transperity)

        if transperity_at_bg:
            transperity = np.where(~tis, 0, transperity)
        else:
            transperity = np.where(~tis, 255, transperity)

        b[:, :, 0] = 0

        # Remap colors
        if color_hg == "red":
            b = b[:, :, ::-1]

        if color_hg == "yellow":
            b = b[:, :, ::-1]
            b[:, :, 2] = b[:, :, 1]
            b[:, :, 1] = b[:, :, 0]

        if color_lg == "green":
            if color_hg == "yellow":
                b[:, :, 1] = b[:, :, 1] + b[:, :, 2]
                b[:, :, 2] = 0
        elif color_lg == "blue":
            if color_hg == "red":
                b[:, :, 2] = b[:, :, 1] + b[:, :, 2]
        elif color_lg == "yellow":
            if color_hg == "red":
                b[:, :, 0] = b[:, :, 0] + b[:, :, 1]

        b[~tis] = color_bg

        # Resize to output size
        b = Image.fromarray(b).resize((width, height),
                                      Image.ANTIALIAS)
        transperity = Image.fromarray(transperity).resize((width, height))

        if grey_tissue:
            img = img.convert('L')
            img = np.array(img)
            img = np.stack((img,) * 3, axis=-1)
            img = Image.fromarray(img)
        img.paste(b, (0, 0), transperity)

        # a = np.where(cm < 85, 0, cm)
        # a = Image.fromarray(a)

        return img

