import cv2

#-------------------------------------------------------------------------------
def parse_video(input_path=None):
    """
    Function to parse a video and return a list of frames
    Args:
        input_path: path to the video file to parse
    Returns:
        frames: list of frames
    """
    video = cv2.VideoCapture(input_path)
    
    if (video.isOpened()== False): 
        print(f"\tError opening video {input_path}")
    else:
        print(f"\tVideo {input_path} opened successfully")
        
    frames = []
    while True:
        ret, frame = video.read()
        if not ret: #ret will be false when there are no more frames to read
            break
        frames.append(frame)
    video.release()
    
    return frames
#-------------------------------------------------------------------------------
def recompose_and_save_video(frames, output_path,fps=30):
    """
    Function to resemble a video from a list of frames and save it
    Args:
        frames: list of frames
        output_path: path to save the video
    """
    height, width, _ = frames[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    
    print(f"\tVideo saved successfully at {output_path}")
#-------------------------------------------------------------------------------