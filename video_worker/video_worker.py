import argparse
import cv2
import time
from datetime import datetime
from threading import Thread

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time

def noThreading(source=0):
    """Grab and show video frames without multithreading."""
    fps = 15
    prev = 0
    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()
    
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break
        while((time.time() - prev) < 1./fps):
          pass
        prev = time.time()
        cv2.imshow("Video", frame)
        cps.increment()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default=0,
        help="Path to video file or integer representing webcam index"
            + " (default 0).")
    ap.add_argument("--thread", "-t", default="none",
        help="Threading mode: get (video read in its own thread),"
            + " show (video show in its own thread), both"
            + " (video read and video show in their own threads),"
            + " none (default--no multithreading)")
    args = vars(ap.parse_args())
    # noThreading(args["source"])
    noThreading("/Users/johntoro/Documents/Projects/TrafficLight/video/source.mp4")