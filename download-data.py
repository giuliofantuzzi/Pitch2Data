#--------------------------------------------------------------------------------
# Libraries
import os
import argparse
import gdown
#--------------------------------------------------------------------------------
# Global variables
DEFAULT_VIDEO_FOLDER_PATH = "input_videos/"

URL_LIST = ["https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF",
            "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
            "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-",
            "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU",
            "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
            ]
#--------------------------------------------------------------------------------
# Main
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Downloader for test videos")
    parser.add_argument("-p", "--path",
                        default=DEFAULT_VIDEO_FOLDER_PATH,
                        help="The path of the folder where to save the videos"
                       )

    args = parser.parse_args()
    path = args.path
    
    if not os.path.exists(path):
        raise IOError(f"Path not found: {path}")
        
    for idx,url in enumerate(URL_LIST):
        video_path = f"{path}video_{idx}.mp4"
        gdown.download(url,video_path)
#--------------------------------------------------------------------------------
