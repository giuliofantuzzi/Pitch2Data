import numpy as np

def crop_image(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Crops the given image based on the given bounding box.

    Args:
        image (np.ndarray): The image to be cropped.
        xyxy (np.ndarray): Bounding box coordinates in format (x1, y1, x2, y2).

    Returns:
        (np.ndarray): The cropped image as a numpy array
    """

    xyxy = np.round(xyxy).astype(int)
    x1, y1, x2, y2 = xyxy
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

def extract_crops(video_frames,tracks,crop_stride=30):
    crops=[]
    for frame in range(0,len(video_frames),crop_stride):
        for _, player_info in tracks['players'][frame].items():
            bbox = player_info['bbox']
            cropped_img = crop_image(video_frames[frame],bbox)  # HxWx3
            crops += [cropped_img]
    return crops
