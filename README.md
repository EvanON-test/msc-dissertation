# FISP Working

Working ML components for integration with hardware

Versions:
- Python 			3.9
- cv2 				4.5.1
- numpy 			1.25.2
- tflite_runtime	2.13.0
- PIL 				10.0.0

### Pipeline.py

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