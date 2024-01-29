from GA import genetic_algorithm, chromosome_to_config
import subprocess
import time
import sys
from measurementAggregator import parseLine, transpose

ORDER = "B-G-L"


def govern(target_latency: float, target_fps: float):
    # Make sure testGovernor exists on the device
    adb_command = "adb shell"

    # Open a subprocess to communicate with ADB shell
    # with open("adb_pipe", "w") as pipe:
    process = subprocess.Popen(adb_command, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr, stdin=subprocess.PIPE, text=True)

    # setup
    process.stdin.write("cd /data/local/Working_dir\n")
    process.stdin.write("export LD_LIBRARY_PATH=/data/local/Working_dir")
    process.stdin.write("echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor")
    process.stdin.write("echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor")
    process.stdin.write("echo 1 > /sys/class/fan/enable")
    process.stdin.write("echo 0 > /sys/class/fan/mode")
    process.stdin.write("echo 4 > /sys/class/fan/level")
    process.stdin.flush()

    adjusted_latency = target_latency
    adjusted_fps = target_fps
    win = False
    while not win:
        pp1, pp2, bfreq, lfreq = chromosome_to_config(genetic_algorithm(10, adjusted_latency, adjusted_fps, 60*2, 50))
        print(f"Trying configuration:\npp1:{pp1}, pp2{pp2}, Big frequency{bfreq}, Small frequency{lfreq}")
        process.stdin.write(f"echo {lfreq} > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq") # little
        process.stdin.write(f"echo {bfreq} > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq") # big
        process.stdin.write(f"./graph_alexnet_all_pipe_sync --threads=4  --threads2=2 --n=60 --total_cores=6 --partition_point={pp1} --partition_point2={pp2} --order={ORDER} &> output.txt")
        process.stdin.write(f"./parse_perf")
        process.stdin.flush()

        try:
            while True:
                # Read the output from the ADB shell
                output = process.stdout.readline().strip()

                # Check if the output is not empty
                if output:
                    # Get the current timestamp at .001 second precision
                    timestamp = time.time()

                    result = f"[{timestamp}] {output}\n"
                    # Print the timestamp and the ADB shell output
                    print(result)
                    _, result = parseLine(result)
                    result = dict(transpose(result))
                    current_fps = result["fps"]
                    current_latency = result["latency"]
                    if current_fps >= target_fps and current_latency <= target_latency:
                        print("Solution found.")
                        win = True
                        return
                    if current_fps < target_fps:
                        adjusted_fps += target_fps - current_fps
                    if current_latency > target_latency:
                        adjusted_latency -= current_fps - target_fps
                    print("Configuration failed to reach performance target.")
                    break

        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C) to stop the script
            print("Script terminated by user.")

        finally:
            # Close the subprocess
            process.terminate()
            break
