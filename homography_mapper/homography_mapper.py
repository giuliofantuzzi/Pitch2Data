# Libraries
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import cv2
import pandas as pd
from mplsoccer import Pitch
import imageio.v3 as iio
from PIL import Image
from io import BytesIO
import os
#import glob
from tqdm import tqdm,trange

# Global variables
# KeyPoints Matrix (w.r.t. 2d pitch in StatsBomb format)
KEYPOINTS_MATRIX = torch.tensor([[0,0],
                                 [0,18],
                                 [0,30],
                                 [0,50],
                                 [0,62],
                                 [0,80],
                                 [6,30],
                                 [6,50],
                                 [12,40],
                                 [18,18],
                                 [18,32],
                                 [18,48],
                                 [18,62],
                                 [60,0],
                                 [60,30],
                                 [60,50],
                                 [60,80],
                                 [102,18],
                                 [102,32],
                                 [102,48],
                                 [102,62],
                                 [108,40],
                                 [114,30],
                                 [114,50],
                                 [120,0],
                                 [120,18],
                                 [120,30],
                                 [120,50],
                                 [120,62],
                                 [120,80],
                                 [50,40],
                                 [70,40],
                                ]
                               )


HOMOGRAPHY_CONF_TRESHOLD = 0.8

PLOT_PITCH_COLORS = {'Players_Home'       : '#EE5252',
                     'Players_Away'       : '#52DDEE',
                     'Players_NotAssigned': '#FFFFFF',
                     'Goalkeepers'        : '#72E585',
                     'Referees'           : '#E9D741',
                    }

# Custom marker for ball
current_dir = os.path.dirname(os.path.abspath(__file__))
svg_path = os.path.join(current_dir, '..', 'utils', 'icons', 'ball.svg')
svg_path = os.path.normpath(svg_path)
ball_path, attributes = svg2paths(svg_path)
BALL_MARKER = parse_path(attributes[0]['d'])
BALL_MARKER.vertices -= BALL_MARKER.vertices.mean(axis=0)
BALL_MARKER = BALL_MARKER.transformed(mpl.transforms.Affine2D().rotate_deg(180))
BALL_MARKER = BALL_MARKER.transformed(mpl.transforms.Affine2D().scale(-1,1))


# Functions
def plt_to_rgb(plot):
    """
    Converts a matplotlib figure to a NumPy array with 3 channels (RGB).

    Args:
        plot: The matplotlib figure object to convert.

    Returns:
        image_np: The plot image as a NumPy array (height, width, 3).
    """
    # Use an in-memory buffer to save the image
    buf = BytesIO()
    plot.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)  # Go to the beginning of the buffer

    # Use PIL to open the image from the buffer
    image = Image.open(buf)

    # Convert the image to an RGB NumPy array
    image_np = np.array(image.convert('RGB'))

    # Close the buffer
    buf.close()
    
    return image_np


class HomographyMapper:
    
    def __init__(self,target=KEYPOINTS_MATRIX, projected_points = None):
        """
        Initializes the HomographyMapper with a target keypoints matrix and optional projected points.
        
        Parameters:
        -----------
        target : torch.Tensor
            The target keypoints matrix to map onto.
        projected_points : dict, optional
            Initial projected points (if available).
        """
        self.target = target
        self.projected_points = None # STILL-TO-DO self.projected_points = self.from_csv(projected_points) if projected_points (from_csv method still to be implemented)
    
    def pitch_homography_(self,keypoints,target):
        """
        Computes the homography matrix given keypoints and target points.

        Parameters:
        -----------
        keypoints : torch.Tensor
            Detected keypoints in the frame.
        target : torch.Tensor
            Target points for mapping.

        Returns:
        --------
        np.ndarray
            The homography matrix.
        """
        # Convert input tensors to numpy (cv2 is numpy friendly) 
        keypoints_np = keypoints.numpy().astype(np.float32)
        target_np = target.numpy().astype(np.float32)
        
        # Find homography matrix
        H, _ = cv2.findHomography(keypoints_np,target_np)
        
        return H
    
    def map_(self,source,homography_matrix):
        """
        Maps source points using the homography matrix.

        Parameters:
        -----------
        source : np.ndarray
            Points to be transformed.
        homography_matrix : np.ndarray
            Homography transformation matrix.

        Returns:
        --------
        np.ndarray
            Mapped points.
        """
        return cv2.perspectiveTransform(source,homography_matrix)

    def map_video(self,video_frames,person_tracks,ball_tracks,keypoints,stretch_dims=None):
        """
        Maps the keypoints of players, referees, and the ball from video frames onto a 2D pitch radar using homography transformation.

        Parameters:
        ----------
        video_frames : list of np.ndarray
            A list of frames from the video, where each frame is represented as a NumPy array.
            
        person_tracks : dict
            A dictionary containing tracking information for players, referees, and goalkeepers. 
            It should have keys 'players', 'referees', and 'goalkeepers', each mapping to a list of dictionaries 
            containing bounding box information for each individual in the respective category.
            
        ball_tracks : list of dict
            A list of dictionaries containing tracking information for the ball across the frames. 
            Each dictionary should include a 'bbox' key for the ball's bounding box.
            
        keypoints : list of Keypoint
            A list of keypoint objects for each frame, where each object is a YOLO keypoint object.
            
        stretch_dims : tuple, optional
            Dimensions to which the video frames was eventually stretched for keypoints detection.

        Returns:
        -------
        projected_points : dict
            A dictionary containing the mapped points for the ball, players, referees, and goalkeepers on the pitch radar.
            The structure is as follows:
            {
                'ball': List of np.ndarray,
                'players': List of dict,
                'referees': List of dict,
                'goalkeepers': List of dict
            }
            Each entry contains the mapped coordinates in 2D space (with NaN for missing data).
        """
        if stretch_dims:
            scale_x = video_frames[0].shape[1] / stretch_dims[1]  # 1920 / 640
            scale_y = video_frames[0].shape[0] / stretch_dims[0]  # 1080 / 640
        else:
            scale_x = 1
            scale_y = 1
        
        projected_points = {'ball'       : [],
                            'goalkeepers': [],
                            'players'    : [],
                            'referees'   : []
                           }
        
        for frame_idx,frame in enumerate(video_frames):

            projected_points['players'].append({})
            projected_points['goalkeepers'].append({})
            projected_points['referees'].append({})
            
            #--------------------------------------------------------------------------------------
            # Extract keypoints
            frame_keypoints = keypoints[frame_idx].keypoints.cpu()
            frame_keypoints_conf = frame_keypoints.conf.squeeze(0).clone()
            frame_keypoints_xy = frame_keypoints.xy.squeeze(0).clone()
            # Filter keypoints with confidence > 0.7                          
            frame_filtered_xy = frame_keypoints_xy[frame_keypoints_conf>HOMOGRAPHY_CONF_TRESHOLD,:]
            # Re-map on original scale
            frame_filtered_xy[:,0]*= scale_x
            frame_filtered_xy[:,1]*= scale_y
            
            
            if frame_filtered_xy.shape[0]>=4: #ensure to have at least 4 keypoints
                #--------------------------------------------------------------------------------------
                # Learn homography matrix for current frame
                frame_target = self.target[frame_keypoints_conf>HOMOGRAPHY_CONF_TRESHOLD,:]
                frame_homography_matrix = self.pitch_homography_(keypoints = frame_filtered_xy,
                                                                 target    = frame_target) #numpy.ndarray
                #--------------------------------------------------------------------------------------
                
                # 1) Map the Ball on the pitch radar
                ball_bbox = ball_tracks[frame_idx].get('bbox',None)
                if ball_bbox:
                    ball_center_x = int((ball_bbox[0]+ball_bbox[2])/2)
                    ball_center_y = int(ball_bbox[3])
                
                projected_points['ball'] += [self.map_(source = np.float32([[ball_center_x,ball_center_y]]).reshape(-1,1,2),
                                      homography_matrix = frame_homography_matrix
                                     )[0,:,:] if ball_bbox else np.array([[np.nan,np.nan]])
                           ]
                
                # 2) Map the Players on the pitch radar
                players_dict = person_tracks['players'][frame_idx]
                for id,player_info in players_dict.items():
                    player_bbox = player_info['bbox']
                    player_center_x = int((player_bbox[0]+player_bbox[2])/2)
                    player_center_y = int(player_bbox[3])
                    
                    
                    projected_points['players'][frame_idx][id] = {'xy' : self.map_(source = np.float32([[player_center_x,player_center_y]]).reshape(-1,1,2),
                                                                               homography_matrix = frame_homography_matrix
                                                                              )[0,:,:],
                                                                  'team' : player_info.get('team',-1) #-1 if team not assigned
                                                             }
                # 3) Map the Refs on the pitch radar
                referees_dict = person_tracks['referees'][frame_idx]
                for id,referee_info in referees_dict.items():
                    referee_bbox = referee_info['bbox']
                    referee_center_x = int((referee_bbox[0]+referee_bbox[2])/2)
                    referee_center_y = int(referee_bbox[3])
                    
                    
                    projected_points['referees'][frame_idx][id] = {'xy' : self.map_(source = np.float32([[referee_center_x,referee_center_y]]).reshape(-1,1,2),
                                                                               homography_matrix = frame_homography_matrix
                                                                              )[0,:,:]
                                                                  }
    
                # 4) Map the Goalkeepers on the pitch radar
                goalkeepers_dict = person_tracks['goalkeepers'][frame_idx]
                for id,goalkeeper_info in goalkeepers_dict.items():
                    goalkeeper_bbox = goalkeeper_info['bbox']
                    goalkeeper_center_x = int((goalkeeper_bbox[0]+goalkeeper_bbox[2])/2)
                    goalkeeper_center_y = int(goalkeeper_bbox[3])
                    
                    
                    projected_points['goalkeepers'][frame_idx][id] = {'xy' : self.map_(source = np.float32([[goalkeeper_center_x,goalkeeper_center_y]]).reshape(-1,1,2),
                                                                               homography_matrix = frame_homography_matrix
                                                                              )[0,:,:]
                                                                  }

            else:

                players_dict     = person_tracks['players'][frame_idx]
                referees_dict    = person_tracks['referees'][frame_idx]
                goalkeepers_dict = person_tracks['goalkeepers'][frame_idx]

                projected_points['ball']                           += [np.array([[np.nan,np.nan]])]
                
                for id,player_info in players_dict.items():
                    projected_points['players'][frame_idx][id]     =  {'xy'   : np.array([[np.nan,np.nan]]),
                                                                       'team' : player_info.get('team',-1)}
                for id,_ in referees_dict.items():
                    projected_points['referees'][frame_idx][id]    =  {'xy' : np.array([[np.nan,np.nan]])}

                for id,_ in goalkeepers_dict.items():
                    projected_points['goalkeepers'][frame_idx][id] =  {'xy' : np.array([[np.nan,np.nan]])}
        
        
        self.projected_points = projected_points
            
        return projected_points

    
    def to_DataFrame(self, csv_path=None):
        """
        Converts the projected points from the video frames into a Pandas DataFrame.

        The DataFrame contains information about the ball, players, goalkeepers, and referees, including their
        positions and associated teams for each frame. The method can also save the DataFrame to a CSV file
        if a path is provided.

        Parameters:
        ----------
        csv_path : str, optional
            The file path where the DataFrame should be saved as a CSV. If provided, the DataFrame will
            be written to this location.

        Raises:
        ------
        RuntimeError
            If homography has not been performed yet (i.e., if `map_video()` has not been called).

        Returns:
        -------
        pd.DataFrame
            A Pandas DataFrame containing the following columns:
            - 'frame': The index of the video frame.
            - 'type': The type of entity ('ball', 'player', 'goalkeeper', or 'referee').
            - 'team': The team associated with the player or goalkeeper (NaN for ball and referees).
            - 'x_coord': The x-coordinate of the entity on the pitch radar.
            - 'y_coord': The y-coordinate of the entity on the pitch radar.
        """

        if not self.projected_points:
            raise RuntimeError('Homography not performed yet. Ensure to call map_video() method first!')

        rows = []

        num_frames = len(self.projected_points['ball'])        # Tanto sono tutte liste di lunghezza uguale

        for frame in range(num_frames):

            ball_info = self.projected_points['ball'][frame]
            rows.append({
                'frame': frame,
                'type': 'ball',
                'team': np.nan,            # No team for the ball
                'x_coord': ball_info[0,0], 
                'y_coord': ball_info[0,1], 
            })

            players_info = self.projected_points['players'][frame]
            rows.extend([{
                'frame': frame,
                'type': 'player',
                'team': players_info[i]['team'],
                'x_coord': players_info[i]['xy'][0,0], 
                'y_coord': players_info[i]['xy'][0,1], 
            } for i in players_info.keys()])

            goalkeepers_info = self.projected_points['goalkeepers'][frame]
            rows.extend([{
                'frame': frame,
                'type': 'goalkeeper',
                'team': np.nan, # We decided to not assign a team to goalkeepers and treat them as a separate entity
                'x_coord': goalkeepers_info[i]['xy'][0,0], 
                'y_coord': goalkeepers_info[i]['xy'][0,1], 
            } for i in goalkeepers_info.keys()])

            referees_info = self.projected_points['referees'][frame]
            rows.extend([{
                'frame': frame,
                'type': 'referee',
                'team': np.nan,
                'x_coord': referees_info[i]['xy'][0,0], 
                'y_coord': referees_info[i]['xy'][0,1], 
            } for i in referees_info.keys()])


        dataframe = pd.DataFrame(rows)
        
        if csv_path:
            dataframe.to_csv(csv_path, index=False)


        return dataframe

    # --------------------------------------------------------------------------------------------------------
    def plot_frame_pitch(self, frame_idx, csv_path=None, img_path=None):
        """
        This method retrieves the data either from a pre-generated CSV file or by converting the 
        projected points into a DataFrame. It then plots entities (players, referees, 
        goalkeepers, and the ball) on a 2d pitch (StatsBomb format). The resulting figure can 
        optionally be saved as an image.

        Parameters:
        -----------
        frame_idx : int
            The index of the frame to plot.
        csv_path : str, optional
            The file path to a CSV file containing position data. If not provided, the data will 
            be generated from the `projected_points` using the `to_DataFrame()` method.
        img_path : str, optional
            The file path where the plotted pitch should be saved as an image. If not provided, 
            the figure will not be saved.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The Matplotlib figure object containing the plot of the pitch and player positions.
        """
        if not csv_path:
            df = self.to_DataFrame()
        else: 
            df = pd.read_csv(csv_path)

        
        pitch= Pitch(pitch_type='statsbomb',
             half=False,
             pitch_color="grass",
             stripe=True,line_color="white")

        fig, ax = pitch.draw(figsize=(10, 4))

        BALL_X = df[(df.frame==frame_idx) & (df.type=='ball')]['x_coord'].values
        BALL_Y = df[(df.frame==frame_idx) & (df.type=='ball')]['y_coord'].values
        
        PLAYERS_X_TEAM0 = df[(df.frame==frame_idx) & (df.type=='player') & (df.team==0)]['x_coord'].values
        PLAYERS_Y_TEAM0 = df[(df.frame==frame_idx) & (df.type=='player') & (df.team==0)]['y_coord'].values
        
        PLAYERS_X_TEAM1 = df[(df.frame==frame_idx) & (df.type=='player') & (df.team==1)]['x_coord'].values
        PLAYERS_Y_TEAM1 = df[(df.frame==frame_idx) & (df.type=='player') & (df.team==1)]['y_coord'].values

        # If clustering is disabled, draw white players
        PLAYERS_X = df[(df.frame==frame_idx) & (df.type=='player') & (df.team==-1)]['x_coord'].values
        PLAYERS_Y = df[(df.frame==frame_idx) & (df.type=='player') & (df.team==-1)]['y_coord'].values
        
        REFS_X = df[(df.frame==frame_idx) & (df.type=='referee')]['x_coord'].values
        REFS_Y = df[(df.frame==frame_idx) & (df.type=='referee')]['y_coord'].values
        
        GOALKEEPERS_X = df[(df.frame==frame_idx) & (df.type=='goalkeeper')]['x_coord'].values
        GOALKEEPERS_Y = df[(df.frame==frame_idx) & (df.type=='goalkeeper')]['y_coord'].values
        
        ax.scatter(PLAYERS_X_TEAM0,PLAYERS_Y_TEAM0,color=PLOT_PITCH_COLORS['Players_Home'],edgecolor="black",s=120,zorder=1)
        ax.scatter(PLAYERS_X_TEAM1,PLAYERS_Y_TEAM1,color=PLOT_PITCH_COLORS['Players_Away'],edgecolor="black",s=120,zorder=1)
        ax.scatter(PLAYERS_X,PLAYERS_Y,color=PLOT_PITCH_COLORS['Players_NotAssigned'],edgecolor="black",s=120,zorder=1)
        ax.scatter(REFS_X,REFS_Y,color=PLOT_PITCH_COLORS['Referees'],edgecolor="black",s=120,zorder=1)
        ax.scatter(BALL_X, BALL_Y, color='white', s=100, marker='o',edgecolor='black',zorder=2) 
        ax.scatter(BALL_X,BALL_Y,marker=BALL_MARKER,s=100,c='black',edgecolor='black',zorder=2)
        ax.scatter(GOALKEEPERS_X,GOALKEEPERS_Y,color=PLOT_PITCH_COLORS['Goalkeepers'],edgecolor="black",s=120,zorder=1)

        fig.tight_layout()
        
        if img_path:
            plt.savefig(img_path)
        plt.close(fig)

        return fig

    def create_gif(self, start_frame=0, end_frame=None, csv_path=None, gif_path='team.gif'):
        """
        Creates a GIF of the football pitch over a sequence of frames, showing the movement of 
        players, referees, goalkeepers, and the ball.

        This method generates individual frame plots using the `plot_frame_pitch()` method and 
        combines them into a GIF. The data for the frames can be provided via a CSV file or 
        generated from `projected_points`. The resulting GIF is saved to the specified path.

        Parameters:
        -----------
        start_frame : int, optional
            The index of the starting frame for the GIF. Default is 0.
        end_frame : int, optional
            The index of the ending frame for the GIF. If not provided, the method automatically 
            uses the last frame available in the data.
        csv_path : str, optional
            The file path to a CSV file containing position data. If not provided, the data will 
            be generated from the `projected_points` using the `to_DataFrame()` method.
        gif_path : str, optional
            The file path where the generated GIF will be saved. Default is 'team.gif'.

        Returns:
        --------
        None
            The method saves the generated GIF to the specified path.
        """
        if not end_frame:
            if not csv_path:
                df = self.to_DataFrame()
            else: 
                df = pd.read_csv(csv_path)
            end_frame = df['frame'].values[-1]


        print(f'>>Making plots...')
        pitch_plots=[self.plot_frame_pitch(i, csv_path) for i in trange(start_frame, end_frame+1)]

        print(f'>>Converting to NumPy...')
        pitch_imgs = [plt_to_rgb(pitch_plot) for pitch_plot in tqdm(pitch_plots)]
        
        print(f'>>Preparing the GIF...')
        iio.imwrite(gif_path, pitch_imgs, duration = 40, loop = 0)

        print(f'\tGIF successfully saved at {gif_path}')

        