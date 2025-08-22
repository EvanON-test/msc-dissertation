# TODO: still not finished. Iterate further when you have completed everything

# FISP Working

Working ML components for integration with hardware

Versions:
- Python 			3.9.18
- cv2 				4.9.0
- numpy 			1.26.4
- tflite_runtime	2.13.0
- PIL 				10.0.0

- NB:(Updated 25/07/25 - For Jetson Nano 2GB Developer Edition)

# pipeline.py - Main file for executing the pipeline.

A Computer Vision pipeline utilising Binary classifier, frame selector, object detector and keypoint detector models over a saved video file.

Default implementation

    Pipeline.run(data_path, monitor=None, runs=1):

Where:
    `data_path`: is the data path of video files to be processed. 
    `monitor`: is the optional monitoring instance (needed for collecting monitoring data)
    `runs`: is the number of runs (needed to loop the single available 30 second video to replicate longer videos for monitoring runs)

Will iterate through each file in the directory until it is finished or is interrupted. 
A file is written to keep track of processed videos in case of interruption and continuation.
This should be called by the catchcam scripts.

Each component in the pipeline has a corresponding utils file for invoking the model / function. 
These are referenced within 'pipeline.py'

## Default Use: `sudo python3.9 pipeline.py`
## Modified Use: `sudo python3.9 pipeline.py --data_path PATHTOVIDEO --runs 6`


# monitoring.py - file for executing the monitoring system.

A monitoring system, used in a threaded approach with the Computer Vision pipeline to benchmark its performance.
Detects hardware (Pi or Nano) and logs their metrics with a default cadence of every 2 seconds as the pipeline runs.

Default implementation

    Monitoring.run(data_path, runs):

Where:
    `data_path`: is the data path of video files to be processed.
    `runs`: is the number of runs (needed to loop the single available 30 second video to replicate longer videos for monitoring runs)

Logs:
    `timestamp`: FORMAT: day - month - year - hour - minute - seconds (Example: 08-07-2025_13-30-31)
    `model_stage`: The specific model running when the metrics are queried 
    `cpu_percent`: percentage of CPU in use
    `ram_percent`percentage of RAM in use
    `cpu_temp`: temperature of CPU
    `gpu_temp`: temperature of GPU (for Nano only)
    `power_used`: amount of power used(Not able to implement without outside hardware)

# TODO:Add more context here once finalised


## Default Use: `sudo python3.9 monitoring.py`
## Modified Use: `sudo python3.9 monitoring.py --data_path PATHTOVIDEO --runs 6`





# realtime_pipeline.py - file for executing the pipeline in real time.

A Computer Vision pipeline utilising Object detector and Keypoint detector models in real time.
Utilises a multi-threaded approach to allow for display (at 30fps) and processing of frames concurrently

Default implementation

    RealtimePipeline.run(process_every_n_frames=60):

Where:
    `process_every_n_frames`: is the cadence at which frames should be processed (to mitigate for performance issues)

 2
# TODO:Add more context here once finalised


## Default Use: `sudo python3.9 realtime_pipeline.py`
## Modified Use: `sudo python3.9 realtime_pipeline.py --frames_interval 120`