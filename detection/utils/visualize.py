import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
import gc
import cv2

from ..detector import Detector
from ..loader import MoviePy as Loader
from ..monitor import Monitor
from ..monitor.movement import assign_events_to_frames, count_events_by_frame
from ..tracker import Tracker
from ..utils import progressbar
from ..visualizer import Visualizer
from .run import get_jobs


def video_worker(video_job, stride, dataset_dir, output_dir,
                 gpu_id, worker_id):
    loader = Loader(video_job.video_name, dataset_dir)
    real_fps = loader.fps / stride
    detector = Detector(gpu_id)
    tracker = Tracker(video_job.video_name, real_fps,
                      **video_job.kwargs.get('tracker_args', {}))
    monitor = Monitor(
        video_job.video_name, real_fps, stride, video_job.video_id,
        video_job.camera_id, loader.video.size[1], loader.video.size[0])
    visualizer = Visualizer()

    loader_iter = loader.read_iter(
        limit=video_job.n_frames, stride=stride, start=video_job.start_frame)
    frames, events = [], []
    print(video_job.n_frames)
    for cnt in range(video_job.n_frames):
        if cnt % 10:
          gc.collect()
        images, image_ids = next(loader_iter)
        frame = detector.detect(images, image_ids)[0]
        frame = tracker.track(frame)
        event = monitor.monit(frame)
        frames.append(frame)
        events.extend(event)
    print("ok", frames)
    event = monitor.finish()
    events.extend(event)
    
    assign_events_to_frames(frames, events)
    event_counts = count_events_by_frame(
        events, len(frames), len(monitor.movements))
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    filename = osp.join(output_dir, '%s_%d-%d-lon.mp4' % (
        osp.splitext(video_job.video_name)[0], video_job.start_frame,
        video_job.start_frame + image_ids[-1] + 1))
    writer = cv2.VideoWriter(
        filename, fourcc, int(loader.fps), tuple(loader.video.size))
    try:
        for frame_i in range(len(frames)):
            visual_image = visualizer.draw_scene(
                frames[frame_i], monitor, event_counts[frame_i])
            frames[frame_i] = None
            writer.write(visual_image[:, :, ::-1])
    finally:
        writer.release()
    return filename


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    n_detector = args.n_gpu * args.n_detector_per_gpu
    jobs = get_jobs(args.video_list_file)
    worker_id = 1
    for job in progressbar(jobs, 'Jobs'):
        video_worker(job, args.stride, args.dataset_dir, args.output_dir,
                    worker_id % args.n_gpu, worker_id)
    print("finished")



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0]))
    parser.add_argument(
        'dataset_dir', help='Path to dataset directory')
    parser.add_argument(
        'video_list_file', help='Path to video list file')
    parser.add_argument(
        'output_dir', help='Path to output directory')
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--n_detector_per_gpu', default=1, type=int)
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_args())
