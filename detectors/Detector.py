# Libraries
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from tqdm import tqdm,trange
import pandas as pd
from abc import ABC, abstractmethod
#-------------------------------------------------------------------------------------------------
# Global variables

TEAM_COLOR_BGR        = {0 : (82,82,238),  # red
                         1 : (238,221,82), # blue
                        } 
PLAYER_COLOR_BGR      = (255,255,255) # white
BALL_COLOR_BGR        = (55,138,233)  # orange
GOALKEEPERS_COLOR_BGR = (133,229,114) # green
REFEREES_COLOR_BGR    = (65,215,233)  # yellow 
KEYPOINTS_COLOR_BGR   = (82, 82, 238) # ~indianred

#-------------------------------------------------------------------------------------------------
# Global functions
def draw_triangle(frame,bbox,color=BALL_COLOR_BGR):
    """
    Draws a filled triangle on the given frame, positioned above the bounding box.
    
    Parameters:
    -----------
    frame : np.ndarray
        The image or frame (in BGR format) on which the triangle will be drawn.
    
    bbox : list
        The bounding box coordinates (x1, y1, x2, y2) where:
        - (x1, y1) is the top-left corner
        - (x2, y2) is the bottom-right corner
    
    color : tuple, optional
        The BGR color value of the triangle. Defaults to `BALL_COLOR_BGR`.

    Returns:
    --------
    np.ndarray
        The modified frame with the triangle drawn on it.
    """
    triangle_center_x = int((bbox[0]+bbox[2])/2)
    triangle_center_y = int(bbox[1]) 
    triangle_height,triangle_width = (20,20)
    triangle_pts = np.array([
        [triangle_center_x,triangle_center_y],
        [triangle_center_x-triangle_width//2,triangle_center_y-triangle_height],
        [triangle_center_x+triangle_width//2,triangle_center_y-triangle_height]
    ]
    )
    cv2.drawContours(frame,[triangle_pts],0,color,cv2.FILLED)
    cv2.drawContours(frame,[triangle_pts],0,(0,0,0),2) #border

    return frame   

def draw_ellipse(frame,bbox,color=PLAYER_COLOR_BGR):
    """
    Draws an ellipse on the given frame, positioned at the bottom of the given bounding box

    Parameters:
    -----------
    frame : np.ndarray
        The image or frame (in BGR format) on which the ellipse will be drawn.
    
    bbox : list
        The bounding box coordinates (x1, y1, x2, y2) where:
        - (x1, y1) is the top-left corner
        - (x2, y2) is the bottom-right corner
    
    color : tuple, optional
        The BGR color value of the ellipse. Defaults to `PLAYER_COLOR_BGR`.

    Returns:
    --------
    np.ndarray
        The modified frame with the ellipse drawn on it.
    """
    # Assume bbox in format x1,y1,x2,y2 where (x1,y1) is the upper-left corner
    ellipse_center_x = int((bbox[0]+bbox[2])/2)
    ellipse_center_y = int(bbox[3])
    ellipse_width = int(bbox[2]-bbox[0])

    # Draw it on the frame
    cv2.ellipse(
        img=frame,
        center = (ellipse_center_x,ellipse_center_y),
        axes = (ellipse_width, int(0.45*ellipse_width)),
        angle = 0.0,
        startAngle = -70,
        endAngle = 245,
        color = color,
        thickness=2,
        lineType = cv2.LINE_4
    )
    return frame
#-------------------------------------------------------------------------------------------------


class Detector(ABC):
    """
    An abstract base class for a YOLO-based detector that processes video frames 
    to detect objects (e.g., players, balls, etc.) using a pre-trained model.

    Attributes:
    -----------
    model : YOLO
        The YOLO object detection model loaded from the provided model path.
    
    Methods:
    --------
    __init__(model_path)
        Initializes the Detector with a YOLO model from the given path.
    
    detections_from_frames_(video_frames, batch_size=30)
        Processes a list of video frames to get object detections in batches.
    
    get_objects_tracks(video_frames)
        Abstract method to detect all the objects across video frames.
    
    annotate_frames(video_frames, tracks)
        Abstract method to annotate objects in video frames based on detections.
    """
    
    def __init__(self,model_path):
        """
        Initializes the Detector class by loading the YOLO model.

        Parameters:
        -----------
        model_path : str
            The path to the YOLO model to be used
        """
        self.model = YOLO(model=model_path) 
    
    def detections_from_frames_(self,video_frames,batch_size=30):
        """
        Detects objects in a list of video frames using the YOLO model.

        This method processes the frames in batches and uses the model to detect
        objects within the frames. The detections are collected and returned as a list.

        Parameters:
        -----------
        video_frames : list of np.ndarray
            A list of video frames to be processed for object detection.
        
        batch_size : int, optional
            The number of frames to process at a time (batch size). Defaults to 30.
            If set to None or 0, all frames will be processed in a single batch.

        Returns:
        --------
        list
            A list of detections where each element corresponds to the detections 
            for a given frame.
        """
        detections = []
            
        if not batch_size:
            batch_size = len(video_frames)
        
        for batch_idx in trange(0,len(video_frames),batch_size,desc='Getting detections'):
            batch = video_frames[batch_idx:(batch_idx+batch_size)]
            batch_detections = self.model.predict(source=batch,
                                                  conf=0.7,
                                                  save = False,
                                                  verbose=False
                                                 )
            detections += batch_detections
            
        return detections

    @abstractmethod
    def get_objects_tracks(self,video_frames):
        pass
        
    @abstractmethod
    def annotate_frames(self,video_frames,tracks):
        pass
#-------------------------------------------------------------------------------------------------

class BallDetector(Detector):
    """
    A class for detecting and tracking the ball in video frames using a YOLO model.

    This class extends the `Detector` class, specifically focusing on detecting 
    a ball and tracking its position over multiple frames. It also provides an 
    option to interpolate missing ball positions when necessary.

    Methods:
    --------
    __init__(model_path)
        Initializes the BallDetector with a YOLO model for ball detection.
    
    get_objects_tracks(video_frames, interpolate_ball=False)
        Detects the ball across video frames, with optional parameter for ball interpolation.
    
    annotate_frames(video_frames, tracks)
        Annotates the ball's position on the video frames
    """
    
    def __init__(self,model_path):
        """
        Initializes the BallDetector class by loading the YOLO model.

        Parameters:
        -----------
        model_path : str
            The path to the YOLO model to be used
        """
        super().__init__(model_path)

    def get_objects_tracks(self,video_frames, interpolate_ball=False):
        """
        Detects the ball across a list of video frames.

        This method uses the YOLO model to detect the ball in each frame and stores
        the bounding box and confidence score. If multiple detections occur, it selects
        the detection with the highest confidence score. Additionally, it can compute 
        missing ball positions across frames through interpolation.

        Parameters:
        -----------
        video_frames : list of np.ndarray
            A list of video frames to process for ball detection.
        
        interpolate_ball : bool, optional
            If True, interpolates the ball's position in frames where the ball is not detected. 
            Default is False.

        Returns:
        --------
        list of dict
            A list where each element corresponds to a frame, and contains a dictionary with
            the ball's bounding box and confidence score (if the ball is detected in that frame).
        """
        
        detections = self.detections_from_frames_(video_frames)
        
        ball_tracks = []
        
        # 1st loop over frames
        for frame,detection in enumerate(tqdm(detections,desc='Storing detections info')):
        
            ball_tracks.append({})

            frame_boxes = detection.boxes

            ball_is_detected = frame_boxes.xyxy.numel()
            
            if ball_is_detected:          
                # bbox     = frame_boxes.xyxy[0,:].squeeze().tolist()  #[0,:] nel caso in cui becchi 2 palloni..tengo solo 1
                # conf     = frame_boxes.conf[0].item() # [0 per los stesso motivo]

                # NB: cosi facendo però scelgo sempre il 1 di potenzialmente piu detection senza valutare la confidenza:
                # se proprio devo sceglierne solo una, a quel punto prendo quella con la confidenza massima...
                # direi che sto approccio è meglio rispetto ad una media delle x e y nel caso di multiple detections nello stesso frame

                conf = frame_boxes.conf

                max_conf, idx = torch.max(conf, dim=0)
                
                ball_tracks[frame]['bbox'] = frame_boxes.xyxy[int(idx.item()),:].squeeze().tolist()
                ball_tracks[frame]['conf'] = max_conf.item()        

        if interpolate_ball: 
            # TO-DO: se non c'è nessun dato per la palla non si puo interpolare...lanciare eccezione!
            
            # Create a dataframe with the informations of the ball
            df = pd.DataFrame(data    = [item.get('bbox',[]) for item in ball_tracks],
                              columns = ['x1','y1','x2','y2'],
                              dtype   = float)

            # Interpolate the empty rows and fill the initial ones
            df.interpolate(inplace=True)
            df.bfill(inplace=True)

            # Update the tracks
            for frame, item in enumerate(ball_tracks):
                if not item:
                    ball_tracks[frame]['bbox'] = df.loc[frame,:].tolist()

                    
        return ball_tracks


    def annotate_frames(self,video_frames,tracks):
        """
        Annotates the ball's position on the video frames using the tracked information.

        This method draws a triangle on each frame to indicate the ball's location,
        based on the tracks returned by `get_objects_tracks`.

        Parameters:
        -----------
        video_frames : list of np.ndarray
            A list of video frames to be annotated.
        
        tracks : 
            The output of get_objects_tracks method

        Returns:
        --------
        list of np.ndarray
            A list of annotated video frames, with the ball's position marked by a triangle.
        """
        # output list with annotated frames
        annotated_frames = []
        
        for frame_idx, frame in enumerate(tqdm(video_frames,desc='Annotating ball')):
            frame= frame.copy()
            # Get dictionaries for the current frame
            ball_dict = tracks[frame_idx]

            # Annotate ball
            if ball_dict: #if dictionary is not empty (ball might have not been detected/interpolated)
                frame = draw_triangle(frame = frame,
                                      bbox  = ball_dict['bbox'],
                                      color = BALL_COLOR_BGR
                                     )
                
            annotated_frames.append(frame)
        return annotated_frames

#-------------------------------------------------------------------------------------------------

class PersonDetector(Detector):
    """
    A class for detecting persons (players, goalkeepers, and referees) in video frames 
    using a YOLO model. This class extends the `Detector` class to specialize in identifying
    and  annotating human objects on the field.

    Methods:
    --------
    __init__(model_path)
        Initializes the PersonDetector with a YOLO model for detecting persons.
    
    get_objects_tracks(video_frames)
        Detects players, goalkeepers, and referees across video frames.
    
    annotate_frames(video_frames, tracks)
        Annotates the detected persons (players, referees, goalkeepers) on video frames.
    """
    
    def __init__(self,model_path):
        """
        Initializes the PersonDetector class by loading the YOLO model for detecting persons.

        Parameters:
        -----------
        model_path : str
            The path to the YOLO model to be used for detecting persons (players, goalkeepers, referees).
        """
        super().__init__(model_path)

    def get_objects_tracks(self,video_frames):
        """
        Detects and tracks players, goalkeepers, and referees across video frames.

        This method processes the video frames using the YOLO model, and for each frame, it detects 
        persons categorized as players, goalkeepers, and referees. The method stores the bounding 
        box and confidence score for each detected object

        Parameters:
        -----------
        video_frames : list of np.ndarray
            A list of video frames to process for person detection.
        
        Returns:
        --------
        dict
            A dictionary containing detections for 'players', 'goalkeepers', and 'referees'. Each key 
            maps to a list where each element corresponds to a frame, and contains a dictionary with 
            object bounding box and confidence score as values.
        """
        
        detections = self.detections_from_frames_(video_frames)
        
        tracks = {'goalkeepers': [],
                  'players'    : [],
                  'referees'   : []
                 }

        # 1st loop over frames
        for frame,detection in enumerate(tqdm(detections,desc='Storing detections info')):
            frame_classes = detection.names
            
            tracks['goalkeepers'].append({})
            tracks['players'].append({})
            tracks['referees'].append({})
        
            # 2nd loop over detected objects in the current frame
            for id,obj in enumerate(detection.boxes):

                #track_id = int(obj.id.item())
                bbox     = obj.xyxy.squeeze().tolist()
                conf     = obj.conf.item()
                cls      = obj.cls.item()
                

                # Add players
                if frame_classes[cls]=='player':
                    tracks['players'][frame][id] = {'bbox' : bbox,
                                                    'conf' : conf 
                                                   }
                # Add goalkeepers
                if frame_classes[cls]=='goalkeeper':
                    tracks['goalkeepers'][frame][id] = {'bbox' : bbox,
                                                        'conf' : conf
                                                       }                                     
                # Add referees
                if frame_classes[cls]=='referee':
                    tracks['referees'][frame][id] = {'bbox' : bbox,
                                                     'conf' : conf
                                                    }
                    
        return tracks

    def annotate_frames(self,video_frames,tracks):
        """
        Annotates players, referees, and goalkeepers on video frames using the detected tracks.

        This method takes the bounding boxes and tracks for players, referees, and goalkeepers, and 
        annotates each frame accordingly. Players are annotated with team-specific colors if clustering
        has been performed beforehand. If no team assignment exists, players are colored white by default.
        Referees and goalkeepers are colored with their specific color (e.g. yellow and green).

        Parameters:
        -----------
        video_frames : list of np.ndarray
            A list of video frames to annotate.
        
        tracks : dict
            The output of get_objects_tracks method

        Returns:
        --------
        list of np.ndarray
            A list of video frames annotated with players, referees, and goalkeepers.
        """
        # output list with annotated frames
        annotated_frames = []
        
        for frame_idx, frame in enumerate(tqdm(video_frames,desc='Annotating persons')):
            
            frame= frame.copy()
            
            # Get dictionaries for the current frame
            players_dict = tracks['players'][frame_idx]
            referees_dict = tracks['referees'][frame_idx]
            goalkeepers_dict = tracks['goalkeepers'][frame_idx]
                
            
            # Annotate Players
            for player_info in players_dict.values():
                player_team = player_info.get('team',-1)
                frame = draw_ellipse(
                    frame    = frame,
                    bbox     = player_info['bbox'],
                    color    = TEAM_COLOR_BGR[player_team] if player_team != -1 else (255,255,255) #if no team detected, white ellipse
                )
            # Annotate Referees
            for referee_info in referees_dict.values():
                frame = draw_ellipse(
                    frame    = frame,
                    bbox     = referee_info['bbox'],
                    color    = REFEREES_COLOR_BGR
                )
            # Annotate Referees
            for goalkeeper_info in goalkeepers_dict.values():
                frame = draw_ellipse(
                    frame    = frame,
                    bbox     = goalkeeper_info['bbox'],
                    color    = (0,255,0) #BGR yellow  
                )
                
            annotated_frames.append(frame)
            
        return annotated_frames

#-------------------------------------------------------------------------------------------------

class KeypointsDetector(Detector):
    """
    A class for detecting and annotating keypoints in video frames using a YOLO-based model. 
    This class extends the `Detector` class to specialize in detecting pitch keypoints
    (see README for details).

    Methods:
    --------
    __init__(model_path)
        Initializes the KeypointsDetector with a YOLO model for detecting keypoints.
    
    get_objects_tracks(video_frames, stretch_dims=None)
        Detects keypoints in video frames, stretching frames if specified.
    
    annotate_frames(video_frames, tracks, stretch_dims=None)
        Annotates the detected keypoints on the original video frames, eventually remapping
        them from stretched dimensions to the original scale.
    """
    
    def __init__(self,model_path):
        """
        Initializes the KeypointsDetector class by loading the YOLO model for detecting keypoints.

        Parameters:
        -----------
        model_path : str
            The path to the YOLO model to be used for keypoint detection.
        """
        super().__init__(model_path)

    def get_objects_tracks(self,video_frames,stretch_dims=None):
        """
        Detects keypoints in the provided video frames, stretching frames if specified.

        If `stretch_dims` is provided, the video frames will be resized to the specified dimensions 
        before running the YOLO model for keypoint detection. Resized dimensions might lead to 
        better performance, particularly (640, 640) on our dataset. Notice that Keypoints are returned
        relative to the stretched dimensions if resizing was applied.

        Parameters:
        -----------
        video_frames : list of np.ndarray
            A list of video frames to process for keypoint detection.
        
        stretch_dims : tuple of int, optional
            The dimensions to which the video frames will be resized for keypoint detection. 
            If None, no resizing is performed. Default is None, but (640, 640) yields better performance 
            in our tests.

        Returns:
        --------
        list
            A list of detections containing the detected keypoints for each frame.
        """
        if stretch_dims:
            stretched_video_frames = [cv2.resize(frame,stretch_dims) for frame in video_frames]
            detections = self.detections_from_frames_(stretched_video_frames,50)
        else:
            detections = self.detections_from_frames_(video_frames,50)
        
        return detections

    def annotate_frames(self,video_frames,tracks,stretch_dims=None):
        """
        Annotates the detected keypoints onto the provided video frames.

        This method takes the detected keypoints and draws them as circles on the original video frames.
        If the frames were resized (stretched) during detection, the keypoints are rescaled to match
        the original frame dimensions.

        Parameters:
        -----------
        video_frames : list of np.ndarray
            A list of video frames to annotate with the detected keypoints.

        tracks : list
            A list of detections containing the keypoints for each frame.

        stretch_dims : tuple of int, optional
            The dimensions to which the video frames were resized for keypoint detection.
            If `None`, the original frame dimensions are used for annotation.
            Default is `None`.

        Returns:
        --------
        list
            A list of annotated frames with detected keypoints drawn as circles.
        """
        if stretch_dims:
            scale_x = video_frames[0].shape[1] / stretch_dims[1]  # 1920 / 640
            scale_y = video_frames[0].shape[0] / stretch_dims[0]  # 1080 / 640
        else:
            scale_x = 1
            scale_y = 1

        annotated_frames=[]

        for frame_idx, frame_tracks in enumerate(tqdm(tracks,desc='Annotating keypoints')):
        
            annotated_frame = video_frames[frame_idx].copy()
            
            frame_keypoints = frame_tracks.keypoints
            if frame_keypoints.xy.shape[1]:
                frame_keypoints_conf = frame_keypoints.conf.squeeze(0).clone()
                frame_keypoints_xy = frame_keypoints.xy.squeeze(0).clone()
                frame_filtered_xy = frame_keypoints_xy[frame_keypoints_conf>0.7,:]
                # Re-map on original scale
                frame_filtered_xy[:,0]*= scale_x
                frame_filtered_xy[:,1]*= scale_y
                for kp in frame_filtered_xy:
                    x,y = int(kp[0]),int(kp[1])
                    cv2.circle(annotated_frame,(x,y), radius=5, color=KEYPOINTS_COLOR_BGR, thickness=-1)
                    cv2.circle(annotated_frame,(x,y), radius=5, color=(0, 0, 0), thickness=2) 
        
            annotated_frames += [annotated_frame]

        return annotated_frames
