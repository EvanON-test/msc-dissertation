#Import statements for required modules
import time
import os
import platform
import psutil
import csv
import datetime
from threading import Thread, Event
import argparse

#Import statement for custom pipeline module
from pipeline import Pipeline

#Try bocks for import statements for modules that could cause failure

#jtop is a nano specific library for accessing hardware metrics
try:
    from jtop import jtop
except ImportError:
    jtop = None
#gpiozero provides CPU temperature on the raspberry pi.
try:
    from gpiozero import CPUTemperature
except ImportError:
    CPUTemperature = None




class BaseMonitor(Thread):
    """Base class of the hardware monitoring classes. It handles the functions common to both of the edge
    devices and inherits Thread in an aid to run as part of a separate thread to the processing"""

    def __init__(self, output_file):
        """Initialises the base monitoring system"""
        #constructor of the parent Thread class
        super().__init__()
        #outputfile for csv logging of metrics
        self.output_file = output_file
        #hardcoded interval (in seconds), to be used as time between each log entry
        self.interval = 2
        #creates a threading event that is needed to stop the thread externally
        self.stop_event = Event()
        #current stage allows the tracking of the stage for csv logging, this equates to moe detailed data
        self.current_stage = None



    def run(self):
        """This is the main monitoring loop. It opens the csv, accesses the metrics and writes them into the csvfile"""
        try:
            with open(self.output_file, 'w') as csvfile:
                #Defines the fieldnames, which are to be the column headers in the resultant csv file.
                fieldnames = ['timestamp', 'model_stage','cpu_percent', 'cpu_temp', 'gpu_percent', 'gpu_temp', 'ram_percent', 'power_used']
                #creats a csv writer object and writes fieldnames into the header row to the csv
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                #While loop that runs until stop event and at an interval defined by the pre-initialised interval value
                while not self.stop_event.wait(self.interval):
                    #collects the hardware metrics via the get metrics function
                    hardware_metrics = self.get_metrics()
                    #Adds the current timestamp as a value to timestamp key
                    hardware_metrics['timestamp'] = time.strftime("%d-%m-%Y_%H-%M-%S")
                    #Adds the current model stage as a value to model_stage key
                    hardware_metrics['model_stage'] = self.current_stage
                    #writes the singular metric run ot the csv file in one row. Iterating each key value pair of the dictionary
                    log_data = hardware_metrics
                    writer.writerow(log_data)
        except Exception as e:
            print("Error occurred in the monitoring thread as: " + str(e))

    def stop(self):
        """Defines the stop function for the monitoring thread (by breaking the while loop in the run function)"""
        self.stop_event.set()

    def get_metrics(self):
        """Gathers metrics that have common access approaches in both devices"""
        #Initilises an empty dictionary
        metrics = {}
        # Adds the current CPU utilisation percentage as a value to cpu_percent key
        metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
        # Adds the current RAM utilisation percentage as a value to ram_percent key
        memory = psutil.virtual_memory()
        metrics['ram_percent'] = memory.percent
        #Returns metrics dictionary
        return metrics




class PiMonitor(BaseMonitor):
    """Raspberry Pi specific implementation of the BaseMonitor class. Extending it with hardware specific functionality"""

    def get_metrics(self):
        """Inherits the capabilities of the parent class. Extends with a pi specific implementation to gather temperatures"""
        #Gets the common metrics
        metrics = super().get_metrics()
        #creates CPUTemperature object and gets reading
        cpu_temp_pi = CPUTemperature().temperature
        # Adds the current cpu temp, as celsius and rounded to 1sf, to cpu_temp key
        metrics['cpu_temp'] = round(cpu_temp_pi, 1)
        #Sets the currently non-gatherable metrics to "N/A""
        metrics['gpu_percent'] = "N/A"
        metrics['gpu_temp'] = "N/A"
        metrics['power_used'] = "N/A"
        return metrics



class NanoMonitor(BaseMonitor):
    """Jetson Nano specific implementation of the BaseMonitor class. Extending it with hardware specific functionality"""

    def __init__(self, output_file):
        """Initialises and extends the BaseMonitors init function. Defines jtop object needed for this class"""
        super().__init__(output_file)
        #Creates an instance of a jtop object
        self.jetson = jtop()


    def run(self):
        """Extends the BaseMonitor 'run' class. First starts the jtop service object before calling the parent 'run' function"""
        self.jetson.start()
        super().run()
        #Stops the jtop service object from running once the run has been completed
        if self.jetson:
            self.jetson.close()

    def get_metrics(self):
        """Inherits the capabilities of the parent class. Extends with a pi specific implementation to gather temperatures"""
        # Gets the common metrics
        metrics = super().get_metrics()
        #Gets the nano metrics using the jtop service object
        #and adds the current cpu temp and gpu temp, as celsius, to the cpu_temp and gpu_temp keys respectively
        metrics['cpu_temp'] = self.jetson.temperature.get('CPU').get('temp')
        metrics['gpu_temp'] = self.jetson.temperature.get('GPU').get('temp')
        #Power metrics not possible on this iteration of NVIDIA's device
        metrics['power_used'] = "N/A"
        return metrics


class Monitoring:
    """Main orchestrator class for running the monitoring system. Checks for hardware type before running the pipeline processes """

    def __init__(self, output_dir="benchmark/" ):
        """Initialises the monitoring system"""
        #defines the output directory
        self.output_dir = output_dir


    def run(self, data_path="processing/video", runs=1):
        """Runs the entire monitoring session: Creates filepath, checks platform type, starts monitoring and runs the pipeline process (which is to be monitored)
        before finally stopping the process"""

        os.makedirs(self.output_dir, exist_ok=True)
        #NOTE: The Pi has to query a server for time, thus time and date are not accurate (although it is still increments so the files are chronological regardless)
        creation_time = datetime.datetime.now()
        timestamp = creation_time.strftime("%Y-%m-%d_%H-%M")#Changed this after first working run. Should order files correctly until/if i change the time access approach
        output_file = os.path.join(self.output_dir, timestamp + ".csv")

        #Gets the monitor type and creates the correct monitor object based on whether a pi or nano specific details are returned
        machine = platform.machine()
        if machine == "armv7l":
            monitor = PiMonitor(output_file=output_file)
        elif machine == "aarch64":
            monitor = NanoMonitor(output_file=output_file)
        else:
            print("Unknown platform type")
            monitor = BaseMonitor(output_file=output_file)

        #Starts the monitors background thread (the start function is inherited from parent Thread class)
        monitor.start()

        #Uses the imported Pipelines run function to run the processing on the data with a monitor instance as input
        try:
            pipeline = Pipeline()
            pipeline.run(data_path=data_path, monitor=monitor, runs=runs)
        except Exception as e:
            print("Error occurred in the monitoring thread as: " + str(e))
        finally:
            #Stops the monitor thread after Pipeline has finished, or Pipeline has failed
            monitor.stop()


if __name__ == "__main__":
    #sets up the command line argument parsing
    parser = argparse.ArgumentParser(description='Run a CV pipeline with a monitoring session for a set number of runs')
    parser.add_argument("--data_path", type=str, default="processing/video", help="Path to folder holding video files")
    parser.add_argument("--runs", type=int, default=1 ,help="Number of runs to run the pipeline for")

    # parses argument and executes pipeline
    args = parser.parse_args()
    monitoring = Monitoring()
    monitoring.run(data_path=args.data_path, runs=args.runs)



















































