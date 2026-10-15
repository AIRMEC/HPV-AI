import sys
import os
import yaml
import pyvips
import h5py

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import box
from shapely.geometry import Polygon

from HERA import HERA

def numpy_to_vips(np_img):
    height, width, bands = np_img.shape
    return pyvips.Image.new_from_memory(np_img.tobytes(), width, height, bands, 'uchar')

def predict_attention_matrix(model, feature_flattened):
    # Model input data should be a single bag of properly collated features (see training routine)
    with torch.no_grad():
        logits, Y_prob, A_raw, m = model(feature_flattened)
    return A_raw.cpu().numpy()

def get_display_image_vips(wsi_path):
    if wsi_path.endswith(".tiff"):
        wsi = pyvips.Image.new_from_file(
            wsi_path,
            access='sequential'  # This may improve performance on Shark NFS I/O, which hates random access.
        )
    else:
        wsi = pyvips.Image.new_from_file(
            wsi_path,
            level=0,
            access='sequential'
        )
    print(f"WSI width, height: {wsi.width, wsi.height}")

    wsi_with = wsi.width
    vips_thumb = pyvips.Image.thumbnail(wsi_path, 2000)
    scale_factor = wsi_with / vips_thumb.width
    print(f"Scaled WSI size: {vips_thumb.width}x{vips_thumb.height}")

    return vips_thumb, scale_factor  

def standardize_scores(raw):
    z_scores = (raw - np.mean(raw)) / np.std(raw)
    z_scores_s = z_scores + np.abs(np.min(z_scores))
    z_scores_s /= np.max(z_scores_s)
    return z_scores_s

def scale_rectangles(raw_rect_bounds, scale_factor):
    scale_factor_inv = 1/ scale_factor
    rects = []
    for coords in raw_rect_bounds:
        # reconstruct the rectangles from the bounds using Shapely's box utility function
        minx, miny, maxx, maxy = coords
        rect = box(minx, miny, maxx, maxy)

        # scale the rectangles using the scale factor
        rect_scaled = scale(
            rect, xfact=scale_factor_inv, yfact=scale_factor_inv, zfact=1.0, origin=(0, 0)
        )
        rects.append(rect_scaled)
    return rects

def build_scoremap(src_img, rect_shapes, scores):
    h, w, _ = np.asarray(src_img).shape

    score_map = np.zeros(dtype=np.float32, shape=(h, w))

    # Assign the scores to the buffer for each rectangle.
    for rect, score in zip(rect_shapes, scores):
        minx, miny, maxx, maxy = rect.bounds
        score_map[round(miny) : round(maxy), round(minx) : round(maxx)] = score

    return score_map

def scoremap_to_heatmap(score_map):
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * score_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGBA)
    assert heatmap.dtype == np.dtype("uint8")

    # Adjust the overlay opacity (0 = completely transparent)
    heatmap[..., 3] = 60
    heatmap[np.where(score_map == 0)] = (255, 255, 255, 0)

    assert heatmap.dtype == np.dtype("uint8")
    assert heatmap.shape[2] == 4
    	
    return pyvips.Image.new_from_array(heatmap)

def get_tile(slide, rect):
    minx, miny, maxx, maxy = rect
    tile = slide.read_region(
        (int(minx), int(miny)), 0, (int(maxx - minx), int(maxy - miny))
    )
    return tile

def load_trained_model(device, args):

    with open(args.config_file, "r") as file:
        try:
            config = yaml.safe_load(file)
            print("Loading config file succesful!")
        except yaml.YAMLError as exc:
            print(exc)

    model = HERA(
        input_feature_size=args.input_feature_size,
        precompression_layer=config["HERA"]["precompression_layer"],
        feature_size_comp=config["HERA"]["feature_size_comp"],
        feature_size_attn=config["HERA"]["feature_size_attn"],
        postcompression_layer=config["HERA"]["postcompression_layer"],
        feature_size_comp_post=config["HERA"]["feature_size_comp_post"],
        dropout=True,
        p_dropout_fc=config["HERA"]["p_dropout_fc"],
        p_dropout_atn=config["HERA"]["p_dropout_atn"]
    ).to(device)

    msg = model.load_state_dict(torch.load(args.checkpoint_HERA, map_location=device), strict=True)
    print('HERA', msg)   

    for p in model.parameters():
        p.requires_grad = False
    print(f"Checkpoint loaded")
    model.eval()

    return model

def read_data(slide_id, data_dir):
    full_path = os.path.join(data_dir, f"{slide_id}_features.h5")
    with h5py.File(full_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

    return features, coords

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Predicting attention map for {args.input_slide}")

    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))

    if not os.path.isfile(args.checkpoint_HERA):
        raise Exception(f"checkpoint {args.checkpoint_HERA} is not a file")
    print("loading checkpoints '{}'".format(args.checkpoint_HERA))

    model = load_trained_model(
        device,
        args,
    )

    model = model.to(device)

    display_image, scale_factor = get_display_image_vips(args.input_slide)
    print("Scale factor: ", scale_factor)

    features, coords = read_data(slide_id, args.data_dir)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()

    device = next(model.parameters()).device
    features = features.to(device)    

    A_raw = predict_attention_matrix(model, features)

    assert A_raw.shape[1] == len(coords)
    assert (A_raw.shape[1] == coords.shape[0]), "Number of attention score sets is not the same as the number of tiles in the batch"

    raw_attn = A_raw[0]  # shape (8132,)

    coords = coords.reshape(-1, 4)
    assert coords.shape[1] == 4, f"Expected coords to have 4 elements per row, got shape {coords.shape}"

    scaled_rects = scale_rectangles(coords, scale_factor)
    z_scores = standardize_scores(raw_attn)
    scoremap = build_scoremap(display_image, scaled_rects, z_scores)

     # apply Gaussian blur, kernel size depends on tile size and desired smoothness
    sigma = 180 / 32
    blurred_scoremap = cv2.GaussianBlur(scoremap, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)
    # normalize again to [0,1]
    blurred_scoremap = (blurred_scoremap - blurred_scoremap.min()) / (blurred_scoremap.max() - blurred_scoremap.min())

    overlay = scoremap_to_heatmap(blurred_scoremap)

    print("Display image dimensions: ", display_image.width, display_image.height, "bands:", display_image.bands)
    print("Overlay image dimensions: ", overlay.width, overlay.height, "bands:", overlay.bands)

    # Make sure both images are RGBA
    if display_image.bands == 3:
        display_image = display_image.bandjoin(255)
    if overlay.bands == 3:
        overlay = overlay.bandjoin(255)

    # Composite the images using alpha blending
    overlay_rgb = overlay.extract_band(0, n=3)
    overlay_alpha = overlay.extract_band(3).cast('float') / 255.0
    display_rgb = display_image.extract_band(0, n=3)
    display_alpha = 1.0 - overlay_alpha

    composite = (overlay_rgb.cast('float') * overlay_alpha + display_rgb.cast('float') * display_alpha).cast('uchar')
    composite = composite.bandjoin(255)

    if composite.bands != 4:
        raise ValueError("Composite image does not have 4 bands. Something went wrong.")

    print("Composite image dimensions: ", composite.width, composite.height, "bands:", composite.bands)

    os.makedirs(os.path.join(args.output_dir, f"{slide_id}"), exist_ok=True)
    outpath = os.path.join(args.output_dir, f"{slide_id}/{slide_id}.png")
    print(f"Exporting {outpath}")
    composite.write_to_file(outpath)

    print("Finished Full attention map, now creating top-ten tiles...")
    ## top twenty tiles
    wsi = pyvips.Image.new_from_file(args.input_slide, access="sequential")
    top_indices = np.argsort(raw_attn)[::-1][:20]  # Get indices of the top 10 highest attention tiles

    # Use the scaled_rects to get tile positions
    for i, idx in enumerate(top_indices):
        rect = scaled_rects[idx]

        if isinstance(rect, Polygon):
            minx, miny, maxx, maxy = rect.bounds
        else:
            minx, miny, maxx, maxy = rect

        left = int(minx * scale_factor)
        top = int(miny * scale_factor)
        width = 720
        height = 720

        print(f"Cropping with left={left}, top={top}, width={width}, height={height}")

        # Prevent out-of-bound crops
        if left + width <= wsi.width and top + height <= wsi.height:
            tile = wsi.crop(left, top, width, height)
            outpath = os.path.join(args.output_dir, f"{slide_id}/{slide_id}_top_tile_{i+1}.png")
            print(f"Exporting {outpath}")
            tile.write_to_file(outpath)
        else:
            print(f"Skipping tile {i+1} at coords {left, top, width, height} (out of bounds)")


    print("Finished exporting top 20 attention tiles.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention heatmap generation script")
    parser.add_argument(
        "--input_slide",
        type=str,
        help="Path to input WSI file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory to saved input features",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_HERA",
        type=str,
        help="Attention model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--input_feature_size",
        help="The size of the input features from the feature bags.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--config_file",
        help="Path to config.yaml file",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args)
