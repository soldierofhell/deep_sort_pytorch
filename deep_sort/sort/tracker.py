# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

import logging
logging.basicConfig(level=logging.DEBUG, filename='/content/app.log', filemode='w')

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=0, team_numbers=None):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.sequence_duration = 0
        self.sequence_no = 0
        
        self.matched_numbers = {}
        
        self.team_numbers = team_numbers

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, new_sequence):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """


        if new_sequence:
            self.sequence_duration = 0
            self.sequence_no += 1
        else:
            self.sequence_duration += 1

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections, min_cost = \
            self._match(detections)
        
        logging.debug(f'matches, unmatched_tracks, unmatched_detections: {matches}, {unmatched_tracks}, {unmatched_detections}, {min_cost}')

        # Update track set.
        for idx, match in enumerate(matches):
            track_idx, detection_idx = match
            match_method = 1
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], match_method, detection_idx, min_cost[idx])
            detections[detection_idx].track_id = self.tracks[track_idx].track_id # for supervisely export
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            new_track = True
            for row, match in enumerate(matches):
                track_idx = match[0]
                iou_with_matched = iou_matching.iou(self.tracks[track_idx].to_tlwh(), detections[detection_idx].tlwh[None,:])[0]
                max_with_matched = iou_matching.iou(self.tracks[track_idx].to_tlwh(), detections[detection_idx].tlwh[None,:], method='MAX')[0]
                logging.debug(f'IOU for {self.tracks[track_idx].track_id} and {detection_idx}: {iou_with_matched}, {max_with_matched}') 
                if iou_with_matched > 0.4 or max_with_matched > 0.7:
                    new_track = False
                    break
            if new_track:
                self._initiate_track(detections[detection_idx], detection_idx)
                detections[detection_idx].track_id = self.tracks[-1].track_id # for supervisely export
            else:
                logging.debug(f'for detection {detection_idx}: too close to {self.tracks[track_idx].track_id} (IOU: {iou_with_matched}, {max_with_matched}')

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks] #  if t.is_confirmed()
        logging.debug(f'active_targets: {active_targets}')
        features, targets = [], []
        for track in self.tracks:
            #if not track.is_confirmed():
            #    continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix
        
        def number_cost(tracks, detections, track_indices=None, detection_indices=None):    
            if track_indices is None:
                track_indices = np.arange(len(tracks))
            if detection_indices is None:
                detection_indices = np.arange(len(detections))

            cost_matrix = np.ones((len(track_indices), len(detection_indices)))

            for row, track_idx in enumerate(track_indices):
                if tracks[track_idx].number is None:
                    continue

                number = tracks[track_idx].number
                candidates = np.asarray([detections[i].number for i in detection_indices])                
                
                team_id = tracks[track_idx].team_id
                team_id_candidates = np.asarray([detections[i].team_id for i in detection_indices])
                
                cost_matrix[row, :] = 1 - np.logical_and(number == candidates, team_id == team_id_candidates)

                # todo: dodac warunki na ta sama druzyne, czyli min features

            return cost_matrix 
        
        def confidence_cost(tracks, detections, track_indices=None, detection_indices=None):    
            if track_indices is None:
                track_indices = np.arange(len(tracks))
            if detection_indices is None:
                detection_indices = np.arange(len(detections))

            cost_matrix = np.ones((len(track_indices), len(detection_indices)))

            for row, track_idx in enumerate(track_indices):

                candidates = np.asarray([detections[i].confidence for i in detection_indices])               
               
                cost_matrix[row, :] = 1 - candidates

            return cost_matrix 

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        logging.debug(f'confirmed_tracks: {confirmed_tracks}')
        logging.debug(f'unconfirmed_tracks: {unconfirmed_tracks}')

        distance_metrics = {
            'I': iou_matching.iou_cost,
            'F': gated_metric,
            'N': number_cost,
            'C': confidence_cost
        }
        
        matches, unmatched_tracks, unmatched_detections, min_cost = \
                linear_assignment.new_matching_cascade(
                    distance_metrics,
                    self.tracks, detections)        
           
        return matches, unmatched_tracks, unmatched_detections, min_cost

    def _initiate_track(self, detection, detection_idx):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection_id=detection_idx, sequence_no=self.sequence_no))
        self._next_id += 1
        
    def _match_number(self, track_id):
        
        current_track = self.tracks[track_id]
        candidates_tracks = []
        
        for track in self.tracks:      
            if self.sequence_duration < track.age: # previous sequence
                confidence = []
                for number_dict in track.number_history:
                    if number_dict['number'] == current.track_number and number_dict['team_id'] == current.team_id:
                        confidence.append(number_dict['confidence'])
                candidates_tracks.append({'track_id': track.track_id, 'sequence_no': track_sequence_no, 'mean_confidence': np.array(confidence).mean(), 'detected': len(confidence), 'total': len(number_dict)})
      
        candidate_tracks = [t for t in candidate_tracks if t['mean_confidence']>0.8 and t['detected']>1 and t['detected']/t['all']>0.5]
        
        #for sequence_no in range(self.sequence_no):
        #    
        #    for candidate_track in candidate_tracks:
        #        if candidate_track.sequence_no == sequence_no:
        
    def update_numbers(self):
        
        self.matched_numbers[self.sequence_no] = {}
        
        for track in self.tracks:
            logging.debug(f'sequence_no: {self.sequence_no}, {track.sequence_no}')
            if self.sequence_no == track.sequence_no: # self.sequence_no - 1
                
                detected_numbers = len(track.number_history)
                
                logging.debug(f'detected_numbers: {detected_numbers}')
                
                numbers = {}
                for number_dict in track.number_history:
                    if number_dict['confidence'] is not None:
                        numbers.setdefault(number_dict['team_id'], {}).setdefault(number_dict['number'], []).append(number_dict['confidence'])
                
                logging.debug(f'numbers: {numbers}')
                
                for team_id, number_dict in numbers.items():
                    for number, conf_list in number_dict.items():
                        conf_count = len(conf_list)
                        conf_mean = sum(conf_list)/conf_count
                        if conf_count >= 2 and conf_mean > 0.5 and number in self.team_numbers[team_id]: 
                            matched = self.matched_numbers[self.sequence_no].setdefault(team_id, {}).setdefault(number, {})                                
                            if matched == {} or matched['score'] < conf_mean:
                                matched = {'track_id': track.track_id, 'score': conf_mean}        
                

                    

