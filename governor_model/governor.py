from GA import genetic_algorithm, chromosome_to_config
import subprocess
import time
import sys
from measurementAggregator import parseLine, transpose

ORDER = "B-G-L"
NUDGE = 1.3
REPETITION_LIMIT = 4

def govern(target_latency: float, target_fps: float):
    # Make sure testGovernor exists on the device
    adb_command = "adb shell"

    # Open a subprocess to communicate with ADB shell
    # with open("adb_pipe", "w") as pipe:
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

    adjusted_latency = target_latency
    adjusted_fps = target_fps
    win = False
    warm = None
    open("warmstart.txt", "w").close()
    # print("hi")
    last_attempt = ''
    stale_count = 0

    while not win:
        c = genetic_algorithm(100, adjusted_latency, adjusted_fps, 30, 40, save="force", warm=warm, save_location="warmstart.txt")
        if str(c) == last_attempt:
            stale_count += 1
        last_attempt = str(c)
        pp1, pp2, bfreq, lfreq = chromosome_to_config(c)
        print(f"\nTrying configuration:\npp1:{pp1}, pp2:{pp2}, Big frequency:{bfreq}, Small frequency:{lfreq}\n")
        process.stdin.write(f"echo {lfreq} > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq\n") # little
        process.stdin.write(f"echo {bfreq} > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq\n") # big
        # process.stdin.write("echo hi\n")
        # process.stdin.flush()
        # print(process.stdout.readline().strip())
        process.stdin.write(f"./graph_alexnet_all_pipe_sync --threads=4  --threads2=2 --n=20 --total_cores=6 --partition_point={pp1} --partition_point2={pp2} --order={ORDER} &> output.txt\n")
        process.stdin.write(f"./parse_perf\n")
        process.stdin.flush()

        try:
            while True:
                # Read the output from the ADB shell
                output = process.stdout.readline().strip()
                print("output is:", output)

                # Check if the output is not empty
                if output:
                    # Get the current timestamp at .001 second precision
                    timestamp = time.time()

                    result = f"[{timestamp}] {output}\n"
                    _, result = parseLine(result)
                    result = dict(transpose(result))
                    current_fps = result["fps"]
                    current_latency = result["latency"]
                    if current_fps >= target_fps and current_latency <= target_latency:
                        print("Solution found.")
                        win = True
                        process.terminate()
                        return
                    nudged = ""
                    if current_fps < target_fps:
                        adjusted_fps += (target_fps - current_fps) * NUDGE
                        nudged = "FPS"
                    if current_latency > target_latency:
                        adjusted_latency -= (current_latency - target_latency) * NUDGE
                        nudged = "latency"
                    warm = "warmstart.txt"

                    print(f"Configuration failed to reach {nudged} target.\n")
                    break

        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C) to stop the script
            print("Script terminated by user.")
            process.terminate()
            break

        if stale_count >= REPETITION_LIMIT:
            print("Governor can't find a better configuration.")
            process.terminate()
            break

    print("bye!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("requires target latency and fps", file=sys.stderr)
        exit(0)
    try:
        target_latency = float(sys.argv[1])
        target_fps = float(sys.argv[2])
    except ValueError:
        print("targets must be numbers",  file=sys.stderr)
    #          lat          fps
    govern(target_latency, target_fps)
