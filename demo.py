from joblib import Parallel, delayed
import subprocess

def video_worker1(args):
  subprocess.call(['python', 'video_worker/video_worker.py'])

def video_worker2(args):
  subprocess.call(['python', 'video_worker/video_worker2.py'])

def arduino_worker(args):
  subprocess.call(['python', 'main.py', args])

def run(task, args):
  if task == "vid1":
    video_worker1(args)
  elif task == "vid2":
    video_worker2(args)
  elif task == "ard":
    arduino_worker(args)

def main():
	tasks = {
    "vid1": "", 
    "vid2": "",
    "ard": ""
  }
	Parallel(n_jobs=3)(delayed(run)(task, args) for task, args in tasks.items())

if __name__ == "__main__":
  main()