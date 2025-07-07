# FISP Working

Working ML components for integration with hardware

Versions:
- Python 			3.9
- cv2 				4.5.1
- numpy 			1.25.2
- tflite_runtime	2.13.0
- PIL 				10.0.0

# Pipeline.py

Main file for executing the pipeline.
Can take two arguments. 

### `sudo python3.9 Pipeline.py`

Runs the following code block:

    if __name__ == "__main__":
        pipeline = Pipeline()
        pipeline.run('processing/video')

Where 'processing/video' is the data path of video files to be processed.

The function will iterate through each file in the dir until it is finished or is interrupted. 
A file is written to keep track of processed videos in case of interruption and continuation.
This should be called by the catchcam scripts.

Each component in the pipeline has a corresponding utils file for invoking the model / function. 
These are referenced within 'Pipeline.py'.


# Monitoring.py

Main file for the monitoring capability.

Includes the different Monitor classes (Base, Pi and Nano) as well as the main 
monitoring thread which detects hardware type before running both the monitor processes
and original CV Pipeline processes concurrently.

### `sudo python3.9 Monitoring.py`

Runs the following code block:

    if __name__ == "__main__":
        monitoring = Monitoring()
        monitoring.run('processing/video')

Where 'processing/video' is the data path of video files to be processed.