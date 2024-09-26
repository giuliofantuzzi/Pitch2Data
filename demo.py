#--------------------------------------------------------------------------------
# Libraries and modules
import os
import argparse
from utils import parse_video,recompose_and_save_video
from detectors import PersonDetector,BallDetector,KeypointsDetector
from homography_mapper import HomographyMapper
from team_classifier import TeamClassifier
from utils import crop_image,extract_crops
import torch as th
#--------------------------------------------------------------------------------
# Global (DEFAULT) variables
PERSONS_MODEL_PATH     = "tuning/pitch_detections/runs/detect/train/weights/best.pt"
BALL_MODEL_PATH        = "tuning/ball_detection/runs/detect/train/weights/best.pt"
KEYPOINTS_MODEL_PATH   = "tuning/pitch_keypoints/runs/pose/train/weights/best.pt"
INPUT_VIDEO_PATH       = "input_videos/"
OUTPUT_VIDEO_PATH      = "output_videos/"
VIDEO_EXT              = ".mp4"
DATAFRAMES_PATH        = "dataframes/"
GIFS_PATH              = "gifs/"
#--------------------------------------------------------------------------------

if __name__ == "__main__":

    
    # Parser
    parser = argparse.ArgumentParser(
        description = "Python script to detect players,referees and ball in all frames of a football video"
    )
    
    # Input video
    parser.add_argument("video",
                        help = "Name of the video to process (without extension!)"
                       )
    
    parser.add_argument("-E","--Extension",
                        default = VIDEO_EXT,
                        help    = "Extension of the video(e.g: '.mp4','.mov',...)"
                       )

    # YOLO models
    parser.add_argument("-P","--PersonsModel",
                        default = PERSONS_MODEL_PATH,
                        help = "Path to the model that detects players and referees"
                       )
    parser.add_argument("-B","--BallModel",
                        default = BALL_MODEL_PATH,
                        help = "Path to the model that detects the ball"
                       )
    parser.add_argument("-K","--KeypointsModel",
                        default = KEYPOINTS_MODEL_PATH ,
                        help = "Path to the model that detects the pitch keypoints"
                       ) 

    # Paths
    parser.add_argument("-I","--InputsDir",
                        default = INPUT_VIDEO_PATH,
                        help = "Path of the input videos directory"
                       ) 
    parser.add_argument("-O","--OutputsDir",
                        default = OUTPUT_VIDEO_PATH,
                        help = "Path of the output videos directory"
                       )
    parser.add_argument("-D","--DataframesDir",
                        default = DATAFRAMES_PATH,
                        help = "Path of the dataframe directory"
                       )
    parser.add_argument("-G","--GifsDir",
                        default = GIFS_PATH, 
                        help = "Path of the gifs directory"
                       )

    # Options and annotations
    parser.add_argument("-PC","--PlayersClustering", 
                        action="store_true",
                        help = "Add this flag to assign players to teams"
                       )
    parser.add_argument("-IB","--InterpolateBall",
                        action="store_true", 
                        help = "Add this flag to interpolate missing ball detections"
                       )
    parser.add_argument("-SG","--StoreGif",
                        action="store_true", 
                        help = "Add this flag to save the gif of the pitch radar"
                       )
    parser.add_argument("-AP","--AnnotatePersons",
                        action="store_true",
                        help = "Add this flag to annotate players and referees"
                       )
    parser.add_argument("-AB","--AnnotateBall",
                        action="store_true",
                        help = "Add this flag to annotate the ball"
                       )
    parser.add_argument("-AK","--AnnotateKeypoints",
                        action="store_true", 
                        help = "Add this flag to annotate the pitch keypoints"
                       )

    # Parse the arguments
    args = parser.parse_args()
    
 
    print(">>Instantiating Trackers")
    person_detector    = PersonDetector(model_path = args.PersonsModel)
    ball_detector      = BallDetector(model_path = args.BallModel)
    keypoints_detector = KeypointsDetector(model_path = args.KeypointsModel)
    
    
    print(">>Importing input video")
    video_frames = parse_video(args.InputsDir + str(args.video) + args.Extension)

    print(">>Tracking persons")
    person_tracks = person_detector.get_objects_tracks(video_frames=video_frames
                                                      )
    print(f">>Tracking ball (Interpolation set to {args.InterpolateBall})")
    ball_tracks   = ball_detector.get_objects_tracks(video_frames=video_frames,
                                                     interpolate_ball=args.InterpolateBall
                                                    )
    print(">>Tracking keypoints")
    keypoints = keypoints_detector.get_objects_tracks(video_frames=video_frames,
                                                      stretch_dims=(640,640)
                                                     )



    if args.PlayersClustering:
        print(">>Clustering part")
        team_classifier = TeamClassifier(device='cuda' if th.cuda.is_available() else 'cpu')
        training_crops = extract_crops(video_frames,person_tracks,crop_stride=30)
        team_classifier.fit(training_crops)
        
        cropped_players = []
        for frame_idx,frame in enumerate(video_frames):
            for _,player_info in person_tracks['players'][frame_idx].items():
                bbox = player_info['bbox']
                cropped_player = crop_image(frame,bbox)  # HxWx3
                cropped_players += [cropped_player]
        
        team_assignment = team_classifier.predict(cropped_players)
        p=0
        for frame_idx,frame in enumerate(video_frames):
            for player_id,player_info in person_tracks['players'][frame_idx].items():
                person_tracks['players'][frame_idx][player_id]['team'] = team_assignment[p]
                p+=1
    
    print(">>Drawing annotations")
    annotated_frames = video_frames.copy()

    # Annotate players, goalkeepers and refs
    if args.AnnotatePersons:
        annotated_frames = person_detector.annotate_frames(video_frames=annotated_frames,
                                                           tracks=person_tracks
                                                          )
    # Annotate Ball (interpolated or not, follwing our above choice)
    if args.AnnotateBall:
        annotated_frames = ball_detector.annotate_frames(video_frames=annotated_frames,
                                                         tracks=ball_tracks
                                                        )
    # Annotate Keypoints
    if args.AnnotateKeypoints:
        annotated_frames = keypoints_detector.annotate_frames(video_frames=annotated_frames,
                                                          tracks=keypoints,
                                                          stretch_dims=(640,640)
                                                         )
    print(">> Saving the annotated video")
    recompose_and_save_video(frames=annotated_frames,
                             output_path=args.OutputsDir +"annotated_"+ str(args.video)+ args.Extension,
                             fps=25
                            )
    # Store GIF
    if args.StoreGif:
        print(">> Saving the GIF")
        homography_mapper = HomographyMapper()
        projected_pts = homography_mapper.map_video(video_frames=video_frames,
                                                    person_tracks=person_tracks,
                                                    ball_tracks=ball_tracks,
                                                    keypoints=keypoints,
                                                    stretch_dims=(640,640))
        
        df = homography_mapper.to_DataFrame(csv_path=args.DataframesDir+ str(args.video)+'.csv')

        homography_mapper.create_gif(start_frame= 0,
                                           end_frame  = None,
                                           csv_path   = args.DataframesDir+str(args.video)+'.csv' \
                                               if os.path.exists(args.DataframesDir+str(args.video)+'.csv') else None,
                                           gif_path   = args.GifsDir+str(args.video)+'.gif')
    
#--------------------------------------------------------------------------------
