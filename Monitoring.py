#Maybe Later
#TODO: Add the whole timings of the run to the csv. (use the basic format already in pipeline)

#Definetly do SOON
#TODO:include actual amount for cpu and ram?
#TODO: tidy comments please


import numpy as np
import time
import cv2
import os

from Pipeline import Pipeline
import platform
import psutil
import csv
import datetime
from threading import Thread, Event
import argparse

#Utilised try bocks to allow for failure, due to the wrong hardware
#jtop is a nano specific library for accessing hardware metrics
try:
    from jtop import jtop
except ImportError:
    jtop = None

#gpiozero provides CPU temperature on the Pi's.
try:
    from gpiozero import CPUTemperature
except ImportError:
    CPUTemperature = None


"""NOTE:I have focused on an inheritance based approach. BaseMonitor is a subclass of the Thread class, thus allowing it the functionality to operate as a new Thread.
PiMonitor and NanoMonitor are subclasses of BaseMonitor thus allowing them to extend the functionality of the BaseMonitor more specifically based on their underlying hardware"""

class BaseMonitor(Thread):
    """Parent of the hardware monitoring classes. It handles the functions common to both of the edge
    devices. It inherits Thread in an aid to run as part of a separate thread to the processing"""
    def __init__(self, output_file):
        #constructor of the parent Thread class
        super().__init__()
        self.output_file = output_file
        #hardcoded interval (in seconds), to be used as time between each log entry
        self.interval = 2
        #creates a threading event that is needed to stop the thread externaly
        self.stop_event = Event()
        self.current_stage = None



    def run(self):
        """Contains the main monitoring loop, opens the csv, accesses the metrics and writes them into the csvfile"""
        try:
            with open(self.output_file, 'w') as csvfile:
                #Defines the column headers in the resultant csv file
                fieldnames = ['timestamp', 'model_stage','cpu_percent', 'cpu_temp', 'gpu_percent', 'gpu_temp', 'ram_percent', 'power_used']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                #This loop will run as long as the stop function has not been called, the wait length is defined by the pre-initialised interval value
                while not self.stop_event.wait(self.interval):
                    hardware_metrics = self.get_metrics()
                    hardware_metrics['timestamp'] = time.strftime("%d-%m-%Y_%H-%M-%S")
                    hardware_metrics['model_stage'] = self.current_stage
                    log_data = hardware_metrics
                    writer.writerow(log_data)
        except Exception as e:
            print("Error occurred in the monitoring thread as: " + str(e))

    def stop(self):
        """Stops the monitoring thread (through breaking the while loop in the Run function)"""
        self.stop_event.set()

    def get_metrics(self):
        """Gathers metrics that have common access approaches in both devices"""
        metrics = {}
        metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        metrics['ram_percent'] = memory.percent
        return metrics




class PiMonitor(BaseMonitor):
    """Child class of the BaseMonitor class. Extends the BaseMonitor class with functionality specific to the Raspberry Pi"""
    def get_metrics(self):
        """Gathers metrics that have common access approaches in both devices and the specific device"""
        #Gets the common metrics
        metrics = super().get_metrics()
        #Uses pi specific approach to get cpu temperature
        cpu_temp_pi = CPUTemperature().temperature
        metrics['cpu_temp'] = round(cpu_temp_pi, 1)
        #TODO: Reinvestigate the options for these later
        #Sets the currently non-gatherable metrics to None
        metrics['gpu_percent'] = "N/A"
        metrics['gpu_temp'] = "N/A"
        metrics['power_used'] = "N/A"
        return metrics



class NanoMonitor(BaseMonitor):
    """Child class of BaseMonitor class. Extends the BaseMonitor class with functionality specific to the Nano."""
    def __init__(self, output_file):
        super().__init__(output_file)
        #Creates an instance of a jtop object (which is needed to access the Nano metrics)
        self.jetson = jtop()


    def run(self):
        """Extends the base run class by first starting the jtop service before calling the BaseMonitor run function"""
        self.jetson.start()
        super().run()
        #Stops the jtop service object from running once the run has been completed
        if self.jetson:
            self.jetson.close()

    def get_metrics(self):
        """Gathers metrics that have common access approaches in both devices and the specific device"""
        # Gets the common metrics
        metrics = super().get_metrics()
        #gets the nano metrics using the jtop service object
        metrics['cpu_temp'] = self.jetson.temperature.get('CPU').get('temp')
        metrics['gpu_temp'] = self.jetson.temperature.get('GPU').get('temp')
        #Power metrics not possible on this iteration of NVIDIA's device
        metrics['power_used'] = "N/A"
        return metrics

#TODO: update this approach to a more general hardware approach (needed if you introduce newer Pi's to the mix)
class Monitoring:
    """Main class for running the benchmark. Checks for hardware type before running the processes """
    @staticmethod
    def platform_type():
        """Detects the hardware type"""
        machine = platform.machine()
        if machine == "armv7l":
            return "pi"
        elif machine == "aarch64":
            return "jetson"
        else:
            return "unknown"

    @staticmethod
    def run(data_path, runs):
        """Runs the entire monitoring session: Creates filepath, checks platform type, starts monitoring and runs the pipeline process (which is to be monitored)
        before finally stopping the process"""
        #Designates the output directory and generates the filename (which is the timestamp of when it is run)
        output_directory = "benchmark/"
        os.makedirs(output_directory, exist_ok=True)
        #NOTE: The Pi has to query a server for time, thus time and date are not accurate (although it is still increments so the files are chronological regardless)
        creation_time = datetime.datetime.now()
        timestamp = creation_time.strftime("%Y-%m-%d_%H-%M")#Changed this after first working run. Should order files correctly until/if i change the time access approach
        output_file = os.path.join(output_directory, timestamp + ".csv")

        #Gets the platform type before checking and then creating teh appropriate monitor object
        platform_type = Monitoring.platform_type()
        monitor = None
        if platform_type == "pi":
            monitor = PiMonitor(output_file=output_file)
        elif platform_type == "jetson":
            monitor = NanoMonitor(output_file=output_file)
        else:
            print("Unknown platform type")

        #Starts the monitors background thread (the start function is inherited from parent Thread class)
        monitor.start()

        #Uses the imported Pipelines run function to run the processing on the data with a monitor instance as input
        try:
            Pipeline.run(data_path=data_path, monitor=monitor, runs=runs)
        except Exception as e:
            print("Error occurred in the monitoring thread as: " + str(e))
        finally:
            #Stops the monitor thread after Pipeline has finished, or Pipeline has failed
            monitor.stop()



if __name__ == "__main__":
    # monitoring = Monitoring()
    # monitoring.run('processing/video')
    #An updated approach. Argparse approach means the number of runs can added to the cli command
    parser = argparse.ArgumentParser(description='Run a CV pipeline with a monitoring session for a set number of runs')
    parser.add_argument("--runs", type=int, default=1 ,help="Number of runs to run the pipeline for")
    args = parser.parse_args()
    Monitoring.run(data_path="processing/video", runs=args.runs)



















































