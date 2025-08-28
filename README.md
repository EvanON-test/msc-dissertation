# FISP Working

Working ML components for integration with Jetson Nano 2GB Developer Edition

Versions:
- Python 			3.9.18
- cv2 				4.9.0
- numpy 			1.26.4
- tflite_runtime	2.13.0
- PIL 				10.0.0

---

# pipeline.py - Main file for executing the pipeline.

A Computer Vision pipeline utilising binary classifier, frame selector, object detector and keypoint detector models over a directory.

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

Default Use:
    
    `sudo python3.9 pipeline.py`
     
Modified Use:

    `sudo python3.9 pipeline.py --data_path PATHTOVIDEO --runs 6`

---

# monitoring.py - Main file for executing the monitoring system.

A monitoring system, gathering hardware metrics in a threaded approach alongside the sequential computer vision pipeline.

Default Implementation:

    Monitoring.run(data_path, runs):


Where:

    `data_path`: is the data path of video files to be processed.
    `runs`: is the number of runs (needed to loop the single available 30 second video to replicate longer videos for monitoring runs)

Will detect the hardware, currently only Pi or Nano, and initialises appropriate monitoring class.
Logs hardware metrics every 2 seconds while pipeline runs alongside it in seperate thread.
Creates timestamped csv files in benchmark directory for performance analysis.


Data this is logged includes:

    `timestamp`: day - month - year - hour - minute - seconds (Example: 08-07-2025_13-30-31)
    `model_stage`: The specific model running when the metrics are queried 
    `cpu_percent`: percentage of CPU in use
    `ram_percent`: percentage of RAM in use
    `cpu_temp`: temperature of CPU
    `gpu_temp`: temperature of GPU (Nano only)
    `power_used`: amount of power used (Not possible on device)


Default Use: 

    `sudo python3.9 monitoring.py`

Modified Use:

    `sudo python3.9 monitoring.py --data_path PATHTOVIDEO --runs 8`


---


# realtime_pipeline.py - Main file for executing the multi-threaded pipeline in real time, headlessly.

A computer vision pipeline utilising motion detection, binary classifier, frame selector, object detector and keypoint detector models
in a multi stage, multithreaded approach on a live camera feed.

Default implementation

    RealtimePipeline.run(process_every_n_frames=60):

Where:

    `process_every_n_frames`: is the cadence at which frames should be processed (to mitigate for performance issues)

Will capture frames from the GStreamer pipeline and detect motion between frames. When motion detected gathers 
30 frames to be analysed by the binary classifier and frame selector in a seperate thread, and then by an object detector
in another thread. High confidence detections spawn another thread that saves and image and keypoint data to csv.

Default Use: 

    `sudo python3.9 realtime_pipeline.py`

Modified Use:

    `sudo python3.9 realtime_pipeline.py --frames_interval 120`

---

# realtime_pipeline_demo.py - Main file for executing the multi-threaded pipeline in real time, with a live display.

A computer vision pipeline utilising motion detection, binary classifier, frame selector, object detector and keypoint detector models
in a multi stage, multithreaded approach on a live camera feed and providing a live display.

Default implementation

    RealtimePipelineDemo.run(process_every_n_frames=60):

Where:

    `process_every_n_frames`: is the cadence at which frames should be processed (to mitigate for performance issues)

Will capture frames from the GStreamer pipeline and overlay information such as the processing status and hardware metrics.
Core processing pipeline is essientially the same as the `realtime_pipeline.py`

Default Use: 

    `sudo python3.9 realtime_pipeline_demo.py`

Modified Use:

    `sudo python3.9 realtime_pipeline_demo.py --frames_interval 120`