
import subprocess
import sys

from time import time

graph = "graph_alexnet_all_pipe_sync"
parts = 8


adb_command = "adb shell"

process = subprocess.Popen(adb_command, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr, stdin=subprocess.PIPE, text=True)

# setup
process.stdin.write("cd /data/local/Working_dir\n")
process.stdin.write("export LD_LIBRARY_PATH=/data/local/Working_dir\n")
process.stdin.write("echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor\n")
process.stdin.write("echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor\n")
process.stdin.write("echo 1 > /sys/class/fan/enable\n")
process.stdin.write("echo 0 > /sys/class/fan/mode\n")
process.stdin.write("echo 4 > /sys/class/fan/level\n")
process.stdin.flush()


# for latency in range(200, 600, 200):
latency = 200
fps = 18

start_time = time()
command = f"./Governor {graph} {parts} {fps} {latency}\n"
print(command)
process.stdin.write(command)
process.stdin.flush()

with open("example_governer_measurements.txt", "a") as file:
    file.write(f"lat={latency}, fps={fps}\n")

while True:
    output = process.stdout.readline().strip()
    print(output)
    if output == "Solution Was Found.":
        result = process.stdout.readline().strip()

        time = time() - start_time
        
        with open("example_governer_measurements.txt", "a") as file:
             file.write(f"{result}, {time}\n\n")
             


        break

    elif output == "No Solution Found":
        time = time() - start_time

        with open("example_governer_measurements.txt", "a") as file:
             file.write(f"no solution, {time}\n\n")
        break

