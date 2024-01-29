# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
# from . import linear_assignment
from . import iou_matching
from .match import Match
from scipy.optimize import linear_sum_assignment as linear_assignment

class Matcher:
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

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.tracks_1 = []
        self.tracks_2 = []
        self._next_id = 1
        self.track_id_map = {}
        self.track_id_map1 = {}
        self.track_id_map2 = {}

        # Additional Feature
        self.track_id_map2_saved = {}  # 存储 map2 的键值对
        self.track_id_map2_count = {}  # 存储 map2 的键值对出现次数
        self.track_id_map2_hits = {}  # 存储已经达到 hit 状态的键值对
        self.track_id_map1_hits = {}
        self.track_id_map2_hits_activate = {} # 存储已经达到 hit 状态的键值对的连续状态
        #TODO 解决连续跳变问题
    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Correspond feature to target.
        # active_targets = [t.track_id for t in self.tracks]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)
        
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self.pair_match(detections)

        # Update track set.
        current_track_id_map1 = {}
        current_track_id_map2 = {}
        for track_idx, detection_idx in matches:
            # self.tracks[track_idx].update(
            #     self.kf, detections[detection_idx])
            current_track_id_map1[self.tracks_1[track_idx].track_id.item()] = self.tracks_2[detection_idx].track_id.item()
            current_track_id_map2[self.tracks_2[detection_idx].track_id.item()] = self.tracks_1[track_idx].track_id.item()
        print('previous')
        print(current_track_id_map1)
        print(current_track_id_map2)
        
        # 此逻辑应用于相机Track id跳变的情况，应用LiDAR使之稳定
        # 遍历当前的LiDAR Track
        for track_id_2, track_id_1 in current_track_id_map2.items():
            # 如果上一帧有相同的匹配, 且hits中没有记录
            if self.track_id_map2.get(track_id_2, None) == track_id_1 and self.track_id_map2_hits.get(track_id_2, None) is None:
                # 写入count + 1
                if self.track_id_map2_count.get(track_id_2, None) is None:
                    self.track_id_map2_count[track_id_2] = 1
                else :
                    self.track_id_map2_count[track_id_2] += 1
            # 如果和上一帧有不同的匹配，且上一帧有匹配, 且hits中没有记录
            elif self.track_id_map2.get(track_id_2, None) != track_id_1 and self.track_id_map2.get(track_id_2, None) is not None and self.track_id_map2_hits.get(track_id_2, None) is None:
                # count归零
                self.track_id_map2_count[track_id_2] = 0
            # 如果和上一帧有不同的匹配，且上一帧有匹配, 且hits中有记录
            # elif self.track_id_map2[track_id_2] != track_id_1 and self.track_id_map2.get(track_id_2, None) is not None and self.track_id_map2_hits.get(track_id_2, None) is not None:

            #     self.track_id_map2_hits_activate[track_id_2] -= 1
            #     # 如果hits_activate == 0
            #     if self.track_id_map2_hits_activate[track_id_2] == 0:
            #         # hits_activate归零
            #         self.track_id_map2_hits.pop(track_id_2)
            #         self.track_id_map2_hits_activate.pop(track_id_2)
            # 如果上一帧有相同的匹配, 且hits中有记录
            elif self.track_id_map2.get(track_id_2, None) == track_id_1 and self.track_id_map2_hits.get(track_id_2, None) is not None:
                self.track_id_map2_hits_activate[track_id_2] = 4
            # if track_id_2 not in self.track_id_map2_count:
            #     self.track_id_map2_count[track_id_2] = 1
            
            # 如果连续 3+1 帧都有相同的匹配
            if self.track_id_map2_count.get(track_id_2, None) == 1:
                # 不需要count了
                self.track_id_map2_count.pop(track_id_2)
                # 写入hits
                self.track_id_map2_hits[track_id_2] = track_id_1
                self.track_id_map1_hits[track_id_1] = track_id_2
                # 写入hits_activate
                self.track_id_map2_hits_activate[track_id_2] = 4

        # # 遍历hits_activate
        # for activate_track_id_2 in self.track_id_map2_hits_activate.keys():
        #     # 如果hits_activate中的track id在当前帧的map中没有匹配
        #     if activate_track_id_2 not in current_track_id_map2.keys():
        #         # hits_activate - 1
        #         self.track_id_map2_hits_activate[activate_track_id_2] -= 1
        #         # 如果hits_activate == 0
        #         if self.track_id_map2_hits_activate[activate_track_id_2] == 0:
        #             # hits_activate归零
        #             self.track_id_map2_hits.pop(activate_track_id_2)
        #             # self.track_id_map1_hits.pop(self.track_id_map2_hits[activate_track_id_2])
        #             print('pop')
        #             self.track_id_map2_hits_activate.pop(activate_track_id_2)
        #             # self.track_id_map2_count.pop(activate_track_id_2)

        # 如果之前不归零。在上一帧有匹配到的track1和当前帧不相符的情况下，track_id_2 in current_track_id_map2.keys() 为True，count不归零继而出现bug
        # 遍历 map2 的键值对出现次数字典
        for track_id_2 in self.track_id_map2_count.keys():
            # 如果当前帧的map中没有这个id
            if track_id_2 not in current_track_id_map2.keys():
                # count归零
                self.track_id_map2_count[track_id_2] = 0

        ### 核心
                
        # 对于 tracks_2 中的每个 track
        for track_id_2, track_id_1 in current_track_id_map2.items():
            # 有hits，当前track_id_1和hits的track_id_1匹配不一致，且上一帧有匹配
            if self.track_id_map2_hits.get(track_id_2, None) is not None and self.track_id_map2_hits.get(track_id_2, None) != track_id_1 and self.track_id_map2.get(track_id_2, None) is not None:
                #把当前帧的track id替换成hits中的track id
                current_track_id_map2[track_id_2] = self.track_id_map2_hits[track_id_2]
                for track in self.tracks_1:
                    if track.track_id.item() == track_id_1:
                        track.track_id = np.array(self.track_id_map2_hits[track_id_2])

        print('hits', self.track_id_map2_hits)
                

        # # 对于 tracks_1 中的每个 track
        # for track in self.tracks_1:
        #     track_id_1 = track.track_id.item()
        #     if current_track_id_map1[track_id_1] != self.track_id_map1_hits.get(track_id_1, None) and track_id_1 in self.track_id_map1_hits.keys():
        #         track.track_id = current_track_id_map1[track_id_1]
        #     for track_id_2_hit, track_id_1_hit in self.track_id_map2_hits.items():
        #         if track_id_1 == track_id_2_hit and track_id_1 != track_id_1_hit:
        #             track.track_id = track_id_1_hit
        #TODO 将替换的ID发回相机Tracker，替换相机Tracker中的ID
        # 遍历LiDAR Track当前帧检测
        for idx, track2 in enumerate(self.tracks_2):
            # 如果当前帧的检测在上一帧的map中有映射（有匹配）
            if track2.track_id.item() in self.track_id_map2.keys():
                # 如果当前帧的检测在当前帧的map中没有匹配
                # if current_track_id_map2[track2.track_id.item()] not in current_track_id_map1.keys():
                #     track2.mark_missed()
                # if current_track_id_map2.get(track2.track_id.item(), None) is None:
                #     # 在track1中用LiDAR当前帧的检测，上一帧LiDAR对应相机的Track id，is_restored属性为True，初始化一个新的track
                #     # 并在当前帧的map中添加恢复的track id的映射
                #     self._initiate_track_1(detections[idx], self.track_id_map2.get(track2.track_id.item(), None), True)
                #     current_track_id_map1[self.track_id_map2.get(track2.track_id.item(), None)] = track2.track_id.item()
                #     current_track_id_map2[track2.track_id.item()] = self.track_id_map2.get(track2.track_id.item(), None)

                if current_track_id_map2.get(track2.track_id.item(), None) is None:
                                    track_id_list = [track.track_id.item() for track in self.tracks_1]
                                    if self.track_id_map2[track2.track_id.item()] not in track_id_list:
                                        
                                        # 在track1中用LiDAR当前帧的检测，上一帧LiDAR对应相机的Track id，is_restored属性为True，初始化一个新的track
                                        # 并在当前帧的map中添加恢复的track id的映射
                                        self._initiate_track_1(detections[idx], self.track_id_map2.get(track2.track_id.item(), None), True)
                                        current_track_id_map1[self.track_id_map2.get(track2.track_id.item(), None)] = track2.track_id.item()
                                        current_track_id_map2[track2.track_id.item()] = self.track_id_map2.get(track2.track_id.item(), None)
                    
        print('after')
        print(current_track_id_map1)
        print(current_track_id_map2)

        # 遍历hits_activate
        for activate_track_id_2 in self.track_id_map2_hits_activate.keys():
            # 如果hits_activate中的track id在当前帧的map中没有匹配
            if activate_track_id_2 not in current_track_id_map2.keys():
                # hits_activate - 1
                self.track_id_map2_hits_activate[activate_track_id_2] -= 1
                # 如果hits_activate == 0
                if self.track_id_map2_hits_activate[activate_track_id_2] == 0:
                    # hits_activate归零
                    self.track_id_map2_hits.pop(activate_track_id_2)
                    # self.track_id_map1_hits.pop(self.track_id_map2_hits[activate_track_id_2])
                    print('pop')
                    self.track_id_map2_hits_activate.pop(activate_track_id_2)
                    break
                    # self.track_id_map2_count.pop(activate_track_id_2)

        # for track_idx in unmatched_tracks:
        #     if track_idx in current_track_id_map2.values():

        #     self.tracks_1[track_idx].mark_missed()
        #     #查看pool中的track id上一帧有没有对应
        #     track_id_value = self.tracks_1[track_idx].track_id.item()
        #     if self.track_id_map is not None and self.track_id_map.get(track_id_value, None) is not None:
        #         tr2_id = self.track_id_map.get(track_id_value, None)
        #         # save_track = self.tracks_2[detection_idx]
        #         # save_track.track_id = self.track_id_map[self.tracks_1[track_idx].track_id.item()]
        #         self._initiate_track_1(detections[detection_idx], self.track_id_map[self.tracks_2[tr2_id].track_id.item()])
        #         self.tracks_1.append(np.array(self.tracks_2[detection_idx]))
        #         #调用lidar投影detections[detection_idx]的bbox到图像上
        #         ############################todo！！！！！！！输入输出统一化
        #     else:
        #         continue

        # for detection_idx in unmatched_detections:
        #     self._initiate_track_1(detections[detection_idx])
        #     self._initiate_track_2(detections[detection_idx])
        
        self.track_id_map1 = current_track_id_map1
        self.track_id_map2 = current_track_id_map2
        
        # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)

    def no_detection(self, current_track_id_map2):
        flag = False
        # 遍历hits_activate
        for activate_track_id_2 in self.track_id_map2_hits_activate.keys():
            # 如果hits_activate中的track id在当前帧的map中没有匹配
            if activate_track_id_2 not in current_track_id_map2.keys():
                # hits_activate - 1
                self.track_id_map2_hits_activate[activate_track_id_2] -= 1
                # 如果hits_activate == 0
                if self.track_id_map2_hits_activate[activate_track_id_2] == 0:
                    # hits_activate归零
                    self.track_id_map2_hits.pop(activate_track_id_2)
                    # self.track_id_map1_hits.pop(self.track_id_map2_hits[activate_track_id_2])
                    print('pop')
                    self.track_id_map2_hits_activate.pop(activate_track_id_2)
                    flag = True
                    break
        if flag == True:
            self.no_detection(current_track_id_map2)
    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Match(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
    
    def _matching_cascade(self, distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
        if track_indices is None:
            track_indices = list(range(len(tracks)))
        if detection_indices is None:
            detection_indices = list(range(len(detections)))

        unmatched_detections = detection_indices
        matches = []
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break

            track_indices_l = [
                k for k in track_indices
                if tracks[k].time_since_update == 1 + level
            ]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue

            matches_l, _, unmatched_detections = \
                self.min_cost_matching(
                    distance_metric, max_distance, tracks, detections,
                    track_indices_l, unmatched_detections)
            matches += matches_l
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections

    def min_cost_matching(self,
            distance_metric, max_distance, tracks, detections, track_indices=None,
            detection_indices=None):
        """Solve linear assignment problem.

        Parameters
        ----------
        distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
            The distance metric is given a list of tracks and detections as well as
            a list of N track indices and M detection indices. The metric should
            return the NxM dimensional cost matrix, where element (i, j) is the
            association cost between the i-th track in the given track indices and
            the j-th detection in the given detection_indices.
        max_distance : float
            Gating threshold. Associations with cost larger than this value are
            disregarded.
        tracks : List[track.Track]
            A list of predicted tracks at the current time step.
        detections : List[detection.Detection]
            A list of detections at the current time step.
        track_indices : List[int]
            List of track indices that maps rows in `cost_matrix` to tracks in
            `tracks` (see description above).
        detection_indices : List[int]
            List of detection indices that maps columns in `cost_matrix` to
            detections in `detections` (see description above).

        Returns
        -------
        (List[(int, int)], List[int], List[int])
            Returns a tuple with the following three entries:
            * A list of matched track and detection indices.
            * A list of unmatched track indices.
            * A list of unmatched detection indices.

        """
        # print([track.track_id.item() for track in self.tracks_1])
        # print([track.track_id.item() for track in self.tracks_2])
        # print([detection.tlwh for detection in detections])
        if track_indices is None:
            track_indices = np.arange(len(tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))

        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.

        cost_matrix = distance_metric(
            tracks, detections, track_indices, detection_indices)
        cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
        row_indices, col_indices = linear_assignment(cost_matrix)

        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in col_indices:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in row_indices:
                unmatched_tracks.append(track_idx)
        for row, col in zip(row_indices, col_indices):
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
                # self.track_id_map[track_idx] = detection_idx
                # self.track_id_map[detection_idx] = track_idx

        return matches, unmatched_tracks, unmatched_detections

    def pair_match(self, detections):
        """
        This function is used to match two detections.
        """
        # def gated_metric(tracks, dets, track_indices, detection_indices):
        #     features = np.array([dets[i].feature for i in detection_indices])
        #     targets = np.array([tracks[i].track_id for i in track_indices])
        #     cost_matrix = self.metric.distance(features, targets)

        #     return cost_matrix
    
        # # Split track set into confirmed and unconfirmed tracks.
        # confirmed_tracks_1 = [
        #     i for i, t in enumerate(self.tracks_1) if t.is_confirmed()]
        # unconfirmed_tracks_1 = [
        #     i for i, t in enumerate(self.tracks_1) if not t.is_confirmed()]

        # # Split track set into confirmed and unconfirmed tracks.
        # confirmed_tracks_2 = [
        #     i for i, t in enumerate(self.tracks_2) if t.is_confirmed()]
        # unconfirmed_tracks_2 = [
        #     i for i, t in enumerate(self.tracks_2) if not t.is_confirmed()]
                
        # Associate confirmed tracks using appearance features.
        # matches_a_1 now not paired track id, but a pairing dict
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     self._matching_cascade(
        #         gated_metric, self.metric.matching_threshold, self.max_age,
        #         self.tracks_1, detections, track_indices=None, detection_indices=None)
        

        
        # matches_a_2, unmatched_tracks_a_2, unmatched_detections_2 = \
        #     linear_assignment.matching_cascade(
        #         gated_metric, self.metric.matching_threshold, self.max_age,
        #         self.tracks_2, detections_2, confirmed_tracks_2)
        
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update != 1]
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)
        
        matches, unmatched_tracks, unmatched_detections = \
            self.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks_1,
                detections, track_indices=None, detection_indices=None)
        # matches = matches_a + matches_b
        # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections
    
    def _initiate_track(self, detection, id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Match(
            mean, covariance, id, self.n_init, self.max_age,
            detection.feature))
        
    def _initiate_track_1(self, detection, id, is_restored = False):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks_1.append(Match(
            mean, covariance, id, self.n_init, self.max_age,
            detection.feature))
        if is_restored:
            self.tracks_1[-1].restored = is_restored
        # self._next_id_1 += 1

    def _initiate_track_2(self, detection, id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks_2.append(Match(
            mean, covariance, id, self.n_init, self.max_age,
            detection.feature))
        # self._next_id_2 += 1

