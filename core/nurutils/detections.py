from dataclasses import dataclass
from typing import List 
from supervision.detection.core import Detections
import numpy as np 
from yolox.tracker.byte_tracker import STrack 
from onemetric.cv.utils.iou import box_iou_batch 

@dataclass(frozen=True) 
class BYTETrackerArgs: 
    track_thresh: float = 0.25 
    track_buffer: int = 30  
    match_thresh: float = 0.8 
    aspect_ratio_thresh: float = 3.0 
    min_box_area: float = 1.0 
    mot20: bool = False 
 

def detections2boxes(detections: Detections) -> np.ndarray: 
    return np.hstack(( 
        detections.xyxy, 
        detections.confidence[:, np.newaxis] 
    )) 
 


def tracks2boxes(tracks: List[STrack]) -> np.ndarray: 
    return np.array([ 
        track.tlbr 
        for track 
        in tracks 
    ], dtype=float) 
 
 

def match_detections_with_tracks( 
    detections: Detections,  
    tracks: List[STrack] 
) -> Detections: 
    if not np.any(detections.xyxy) or len(tracks) == 0: 
        return np.empty((0,)) 
 
    tracks_boxes = tracks2boxes(tracks=tracks) 
    iou = box_iou_batch(tracks_boxes, detections.xyxy) 
    track2detection = np.argmax(iou, axis=1) 
     
    tracker_ids = [None] * len(detections) 
     
    for tracker_index, detection_index in enumerate(track2detection): 
        if iou[tracker_index, detection_index] != 0: 
            tracker_ids[detection_index] = tracks[tracker_index].track_id 
 
    return tracker_ids 