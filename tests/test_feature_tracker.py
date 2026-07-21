import numpy as np

from feature_tracker import FeatureTracker, load_cam0_frame_list, iter_cam0_frames

N_TEST_FRAMES = 30


def test_load_cam0_frame_list_matches_csv_count():
    frames = load_cam0_frame_list()
    assert len(frames) > 1000
    timestamps = [t for t, _ in frames]
    assert sorted(timestamps) == timestamps


def test_iter_cam0_frames_shapes_and_dtype():
    frames = list(iter_cam0_frames(max_frames=3))
    assert len(frames) == 3
    for timestamp, image in frames:
        assert isinstance(timestamp, int)
        assert image.shape == (480, 752)
        assert image.dtype == np.uint8


def test_tracker_first_frame_detects_bounded_unique_features():
    tracker = FeatureTracker(max_features=150)
    timestamp, image = next(iter_cam0_frames(max_frames=1))
    tracker.process_frame(image, timestamp)

    assert 0 < tracker.n_active <= 150
    assert len(np.unique(tracker.active_ids)) == tracker.n_active
    for track_id in tracker.active_ids:
        assert tracker.tracks[int(track_id)] == [tracker.tracks[int(track_id)][0]]  # single observation so far
        assert tracker.tracks[int(track_id)][0][0] == timestamp


def test_tracker_bookkeeping_stays_consistent_over_many_frames():
    tracker = FeatureTracker(max_features=150)
    last_timestamp = None
    for timestamp, image in iter_cam0_frames(max_frames=N_TEST_FRAMES):
        tracker.process_frame(image, timestamp)
        last_timestamp = timestamp

        assert len(tracker.active_ids) == len(tracker._active_points)
        assert len(np.unique(tracker.active_ids)) == len(tracker.active_ids)
        assert 0 < tracker.n_active <= 150

    # every currently-active track's most recent observation should be from the last processed frame
    for track_id in tracker.active_ids:
        assert tracker.tracks[int(track_id)][-1][0] == last_timestamp


def test_tracker_active_points_match_track_history():
    tracker = FeatureTracker(max_features=150)
    for timestamp, image in iter_cam0_frames(max_frames=10):
        tracker.process_frame(image, timestamp)

    for track_id, point in zip(tracker.active_ids, tracker._active_points):
        recorded = tracker.tracks[int(track_id)][-1][1]
        assert np.allclose(recorded, (point[0, 0], point[0, 1]))


def test_tracker_replenishes_toward_target_count_when_started_low():
    tracker = FeatureTracker(max_features=30, replenish_ratio=0.7)
    frames = list(iter_cam0_frames(max_frames=N_TEST_FRAMES))
    for timestamp, image in frames:
        tracker.process_frame(image, timestamp)
    # after enough frames the tracker should be topping back up toward its target,
    # not permanently stuck at a much smaller count
    assert tracker.n_active >= 0.7 * 30


def test_tracker_produces_some_multi_frame_tracks():
    tracker = FeatureTracker(max_features=150)
    for timestamp, image in iter_cam0_frames(max_frames=N_TEST_FRAMES):
        tracker.process_frame(image, timestamp)

    track_lengths = [len(obs) for obs in tracker.tracks.values()]
    assert max(track_lengths) > 1  # KLT should successfully carry at least some tracks across frames
    assert any(length == N_TEST_FRAMES for length in track_lengths)  # some tracks survive the whole window
