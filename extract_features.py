import time
import cv2
import h5py
import numpy as np
import openslide
import torch
from PIL import Image, ImageDraw
from shapely.affinity import scale
from shapely.prepared import prep
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pyvips
from multiprocessing import Pool, cpu_count
from huggingface_hub import login

def segment_tissue(img, red_filter):
    img_prepped = img.copy()
    if red_filter:
        print("Applying red filter")
        # UMCG images contain weird all-red background pixels that should first be set to white.
        lower = np.array([1, 1, 1])
        upper = np.array([250, 255, 255])
        thresh = cv2.inRange(img_prepped, lower, upper)
        img_prepped[thresh != 255] = (255,255,255)

    img_hsv = cv2.cvtColor(img_prepped, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy

def detect_foreground(contours, hierarchy):
    print("Now in function detect foreground")
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # find foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours

def construct_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
        if len(foreground) < 3:
            continue

        # remove redundant dimensions from the contour and convert to Shapely Polygon
        poly = Polygon(np.squeeze(foreground))

        # discard all polygons that are considered too small
        if poly.area < min_area:
            continue

        if not poly.is_valid:
            # This is likely becausee the polygon is self-touching or self-crossing.
            # Try and 'correct' the polygon using the zero-length buffer() trick.
            # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
            poly = poly.buffer(0)

        # Punch the holes in the polygon
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue

            hole = Polygon(np.squeeze(hole_contour))

            if not hole.is_valid:
                continue

            # ignore all very small holes
            if hole.area < min_area:
                continue

            poly = poly.difference(hole)

        polys.append(poly)

    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    return unary_union(polys)

def generate_tiles(
    tile_width_pix, tile_height_pix, img_width, img_height, offsets=[(0, 0)]
):
    # Generate tiles covering the entire image.
    # Provide an offset (x,y) to create a stride-like overlap effect.
    # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
    range_stop_width = int(np.ceil(img_width + tile_width_pix))
    range_stop_height = int(np.ceil(img_height + tile_height_pix))

    rects = []
    for xmin, ymin in offsets:
        cols = range(int(np.floor(xmin)), range_stop_width, tile_width_pix)
        rows = range(int(np.floor(ymin)), range_stop_height, tile_height_pix)
        for x in cols:
            for y in rows:
                rect = Polygon(
                    [
                        (x, y),
                        (x + tile_width_pix, y),
                        (x + tile_width_pix, y - tile_height_pix),
                        (x, y - tile_height_pix),
                    ]
                )
                rects.append(rect)
    return rects

def make_tile_QC_fig(tiles, thumb_array, scale_factor, line_width_pix=1, extra_tiles=None):
    img = Image.fromarray(thumb_array)
    draw = ImageDraw.Draw(img, "RGBA")
    downsample = 1 / scale_factor
    for tile in tiles:
        bbox = tuple(np.array(tile.bounds) * downsample)
        draw.rectangle(bbox, outline="lightgreen", width=line_width_pix)

    # allow to display other tiles, such as excluded or sampled
    if extra_tiles:
        for tile in extra_tiles:
            bbox = tuple(np.array(tile.bounds) * downsample)
            draw.rectangle(bbox, outline="blue", width=line_width_pix + 1)

    return img

def create_tissue_mask(wsi_path, wsi, red_filter):
    wsi_with = wsi.width

    # Use libvips to create the thumbnail as OpenSlide can throw OOM errors for non-pyramid images (i.e. images that only have a level 0).
    vips_thumb = pyvips.Image.thumbnail(wsi_path, 2000)

    # Get the total surface area of the slide level that was used
    level_area = vips_thumb.width * vips_thumb.height

    # Minimum surface area of tissue polygons (in pixels)
    # Note that this value should be sensible in the context of the chosen tile size
    min_area = level_area / 500
    
    thumb_array = vips_thumb.numpy()
    contours, hierarchy = segment_tissue(thumb_array, red_filter)

    foreground_contours, hole_contours = detect_foreground(contours, hierarchy)
    tissue_mask = construct_polygon(foreground_contours, hole_contours, min_area)

    # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
    scale_factor = wsi_with / vips_thumb.width
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
    )

    return tissue_mask_scaled, thumb_array, scale_factor    

def create_tissue_tiles(mpp, tissue_mask_scaled, tile_size_microns, offsets_micron=None):
    print(f"tile size is {tile_size_microns} um")
    tile_size_pix = round(tile_size_microns / mpp)
    print(f"tile size is {tile_size_pix} pixels")

    # Use the tissue mask bounds as base offsets (+ a margin of a few tiles) to avoid wasting CPU power creating tiles that are never going
    # to be inside the tissue mask.
    tissue_margin_pix = tile_size_pix * 2
    minx, miny, maxx, maxy = tissue_mask_scaled.bounds
    min_offset_x = minx - tissue_margin_pix
    min_offset_y = miny - tissue_margin_pix
    offsets = [(min_offset_x, min_offset_y)]

    if offsets_micron is not None:
        assert (
            len(offsets_micron) > 0
        ), "offsets_micron needs to contain at least one value"
        # Compute the offsets in micron scale
        offset_pix = [round(o / mpp) for o in offsets_micron]
        offsets = [(o + min_offset_x, o + min_offset_y) for o in offset_pix]

    # Generate tiles covering the entire WSI
    all_tiles = generate_tiles(
        tile_size_pix,
        tile_size_pix,
        maxx + tissue_margin_pix,
        maxy + tissue_margin_pix,
        offsets=offsets,
    )

    # Retain only the tiles that sit within the tissue mask polygon
    filtered_tiles = [rect for rect in all_tiles if tissue_mask_scaled.intersects(rect)]

    return filtered_tiles

def tile_is_not_empty(tile, threshold_white=20):
    histogram = tile.histogram()

    # Take the median of each RGB channel. Alpha channel is not of interest.
    # If roughly each chanel median is below a threshold, i.e close to 0 till color value around 250 (white reference) then tile mostly white.
    whiteness_check = [0, 0, 0]
    for channel_id in (0, 1, 2):
        whiteness_check[channel_id] = np.median(
            histogram[256 * channel_id : 256 * (channel_id + 1)][100:200]
        )

    if all(c <= threshold_white for c in whiteness_check):
        # exclude tile
        return False

    # keep tile
    return True

def crop_rect_from_slide(slide, rect):
    minx, miny, maxx, maxy = rect.bounds #minx, miny: The top-left coordinates of the rectangle; maxx, maxy: The bottom-right coordinates of the rectangle
    width = int(maxx - minx)
    height = int(maxy - miny)

    # Ensure coordinates are within the image bounds
    slide_width, slide_height = slide.width, slide.height #width and height of the entire slide image

    #max(0, ...): Ensures minx is not negative (which would be out of bounds).
    minx = max(0, min(slide_width - 1, int(minx)))  # Ensure minx is within image width (The highest valid x coordinate is slide_width - 1)
    miny = max(0, min(slide_height - 1, int(miny)))  # Ensure miny is within image height
    maxx = max(0, min(slide_width, int(maxx)))       # Ensure maxx is within image width
    maxy = max(0, min(slide_height, int(maxy)))      # Ensure maxy is within image height

    #print(f"Cropping region from coordinates: {minx}, {miny}, {maxx}, {maxy}")
    region = slide.crop(minx, miny, width, height)
    
    return region

#A PyTorch dataset class that provides tiles to the model for feature extraction, resizing them to a specified size and applying transformations.
class BagOfTiles(Dataset):
    def __init__(self, wsi, tiles, transform):
        self.wsi = wsi
        self.tiles = tiles

        self.roi_transforms = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        try:
            img = crop_rect_from_slide(self.wsi, tile)
        except Exception as e:
            # Catch unfixable erors like "openslide.lowlevel.OpenSlideError: Corrupt JPEG data: premature end of data segment" and "openslide.lowlevel.OpenSlideError: Not a JPEG file: starts with 0x00 0x00"
            print(f"Could not read slide region {tile} because of: ", e)
            return None, None

        # Ensure we have a square tile in our hands.
        # We can't handle non-squares currently, as this would require changes to the aspect ratio when resizing.
        width, height = img.width, img.height
        assert width == height, "input image is not a square"

        # Convert PyVips image to NumPy array
        img_np = np.ndarray(buffer=img.write_to_memory(),
                            dtype=np.uint8,
                            shape=(img.height, img.width, img.bands))

        # Convert NumPy array to PIL image (RGB)
        img_pil = Image.fromarray(img_np)

        # Ensure the image is in RGB format (if it's not already)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        # Apply transformations (resize, normalization, etc.)
        img_pil = self.roi_transforms(img_pil)

        coord = tile.bounds
        return img_pil, coord

def collate_features(batch):
    list_images = [item[0] for item in batch if item[0] is not None]
    list_coords = [item[1] for item in batch if item[1] is not None]
    if len(list_images) == 0:
        return [None, None]
    else:
        img = torch.stack(list_images, dim=0)
        coords = np.vstack(list_coords)
    return [img, coords]

def write_to_h5(file, asset_dict):
    for key, val in asset_dict.items():
        if key not in file:
            maxshape = (None,) + val.shape[1:]
            dset = file.create_dataset(
                key, shape=val.shape, maxshape=maxshape, dtype=val.dtype
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + val.shape[0], axis=0)
            dset[-val.shape[0] :] = val

def load_encoder(pretrained_path, device):    
    token_hop1 = '...' #Put your token here
    login(token_hop1, add_to_git_credential=True)
    print("Loading model:", "hf-hub:bioptimus/H-optimus-1")

    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-1",
        init_values=1e-5,
        dynamic_img_size=True,
    )

    e = model.load_state_dict(
        torch.load(
            os.path.join(pretrained_path, "pytorch_model.bin"),
            map_location="cpu",
            #weights_only=True,
        ),
        strict=True,
    )
    print(e)

    model.to(device)
    model.eval()
    return model

def extract_features(model, device, wsi, filtered_tiles, workers, batch_size):
    using_cuda = device.type == "cuda"
    kwargs = (
        {"num_workers": workers, "pin_memory": True} if using_cuda else {}
    )
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
    print(transform)

    loader = DataLoader(
        dataset=BagOfTiles(wsi, filtered_tiles, transform),
        batch_size=batch_size,
        num_workers=workers,  
        collate_fn=collate_features,
        pin_memory=(device.type == "cuda"),  # Ensures faster CPU-to-GPU transfers
        persistent_workers=(workers > 0),  # Keep workers alive for efficiency
        prefetch_factor=2 if workers > 0 else None,  # Preloads next batches into RAM
    )

    # We recommend using mixed precision for faster inference.
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=using_cuda):
        with torch.inference_mode():
            for batch, coords in loader:
                if batch is None:
                    continue
                batch = batch.to(device, non_blocking=True)
                output = model(batch).cpu().numpy()
                assert output.shape == (batch.shape[0], 1536), output.shape
                assert coords.shape == (batch.shape[0], 4), coords.shape
                yield output, coords

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Preprocessing script")

    parser.add_argument(
        "--input_slide",
        type=str,
        help="Path to input WSI file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--path_Hoptimus1_weights",
        type=str,
        required=True,
        help="Path to the pretrained H-optimus-1 weights",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default= -1.0,
    )
    parser.add_argument(
        "--tile_size",
        help="Desired tile size in microns (should be the same value as used in feature extraction model).",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--red_filter",
        help="Whether to apply the tissue extraction 'red background' filter",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
    wip_file_path = os.path.join(args.output_dir, slide_id + "_wip.h5") 
    output_file_path = os.path.join(args.output_dir, slide_id + "_features.h5") 

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(output_file_path):
        raise Exception(f"{output_file_path} already exists")
    
    start_time = time.time()
    red_filter = args.red_filter > 0

    print("Now opening WSI with pyvips")
    wsi = pyvips.Image.new_from_file(
        args.input_slide,
        access='sequential'  # This may improve performance on Shark NFS I/O, which hates random access.
    )
    print(f"WSI width, height: {wsi.width, wsi.height}")

    print("Now creating tissue mask")
    tissue_mask_scaled, thumb_array, scale_factor = create_tissue_mask(args.input_slide, wsi, red_filter)

    try:
        # Attempt to retrieve the image description metadata
        image_desc = wsi.get_value('image-description')
        description = {i.split('=')[0].strip(): i.split('=')[1].strip() 
                       for i in image_desc.split('|')[1:]}
        mpp = float(description['MPP'])
        print('mpp', mpp)
    except Exception as e:
        print("Warning: could not get image-description metadata. Error:", e)
        # Set a default value if metadata is not available.
        mpp = 0.25  # Adjust this default as appropriate
        print(f"Therefore now using default value MPP={mpp}")

    print("Now creating tissue tiles")
    filtered_tiles = create_tissue_tiles(mpp, tissue_mask_scaled, args.tile_size)

    print("Now making QC figure")
    qc_img = make_tile_QC_fig(filtered_tiles, thumb_array, scale_factor, 2)

    # Resize while maintaining aspect ratio
    qc_img_target_width = 1920
    qc_img_target_height = int(qc_img.height * (qc_img_target_width / qc_img.width))
    qc_img = qc_img.resize((qc_img_target_width, qc_img_target_height))

    print(
        f"Finished creating {len(filtered_tiles)} tissue tiles in {time.time() - start_time}s"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    print(f"Device = {device}") 

    model = load_encoder(args.path_Hoptimus1_weights, device=device)

    generator = extract_features(
        model,
        device,
        wsi,
        filtered_tiles,
        args.workers,
        args.batch_size
    )
    start_time = time.time()
    count_features = 0
    with h5py.File(wip_file_path, "w") as file:
        for i, (features, coords) in enumerate(generator):
            count_features += features.shape[0]
            write_to_h5(file, {"features": features, "coords": coords})
            print(
                f"Processed batch {i}. Extracted features from {count_features}/{len(filtered_tiles)} tiles in {(time.time() - start_time):.2f}s."
            )

    os.rename(wip_file_path, output_file_path)

    # Save QC figure while keeping track of number of features/tiles used since RBG filtering is within DataLoader.
    qc_img_file_path = os.path.join(
        args.output_dir, f"{slide_id}_{count_features}_features_QC.png"
    )
    qc_img.save(qc_img_file_path)
    print(
        f"Finished extracting {count_features} features in {(time.time() - start_time):.2f}s"
    )