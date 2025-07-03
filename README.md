# FISP Working

Working ML components for integration with hardware

Versions:
- Python 			3.9
- cv2 				4.5.1
- numpy 			1.25.2
- tflite_runtime	2.13.0
- PIL 				10.0.0


## Pipeline.py - Delete once new version is tested

Main file for executing the pipeline.
Takes one argument that is the directory of video files to be processed. 

`Pipeline.run(data_path)`

The function will iterate through each file in the dir until it is finished or is interrupted. 
A file is written to keep track of processed videos in case of interruption and continuation.
This should be called by the catchcam scripts.

Each component in the pipeline has a corresponding utils file for invoking the model / function. 
These are referenced within 'Pipeline.py'.

The model files required to run the pipeline are too large to be stored on github. 
For access to these files contact the maintainer.

## Updated Pipeline.py

Main file for executing the pipeline.
Can take two arguments: 

### `Pipeline.run(data_path)`

Where data_path is the directory of video files to be processed.

The function will iterate through each file in the dir until it is finished or is interrupted. 
A file is written to keep track of processed videos in case of interruption and continuation.
This should be called by the catchcam scripts.

Each component in the pipeline has a corresponding utils file for invoking the model / function. 
These are referenced within 'Pipeline.py'.

The model files required to run the pipeline are too large to be stored on github. 
(True in the original setup....However not as of 03/07/2025)

### `Pipeline.run(data_path, monitor)`

Where monitor is an instance of a monitor child class from Monitoring.py 
(the monitor parent class being itself a child class of Thread class ) 

## Monitoring.py

Main file for the monitoring capability.

Includes the different Monitor classes (Base, Pi and Nano) as well as the main 
Monitoring thread which detects hardware type and then runs both the monitor process on
a seperate thread and the original Pipeline process which is to be monitored