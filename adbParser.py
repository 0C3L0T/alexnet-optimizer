import subprocess
import time
import sys


def adb_shell_listener():
    # Make sure testGovernor exists on the device
    adb_command = "adb shell"

    # Open a subprocess to communicate with ADB shell
    # with open("adb_pipe", "w") as pipe:
    process = subprocess.Popen(adb_command, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr, stdin=subprocess.PIPE, text=True)

    process.stdin.write("cd /data/local/Working_dir\n")
    process.stdin.write("./governor\n")
    process.stdin.flush()
    try:
        while True:
            # Read the output from the ADB shell
            output = process.stdout.readline().strip()
            # print(f"output: {output}")

            # Check if the output is not empty
            if output:
                # Get the current timestamp at .001 second precision
                timestamp = time.time()

                result = f"[{timestamp}] {output}\n"
                # Print the timestamp and the ADB shell output
                print(result)

                # write the output and timestamp to a file
                with open("governor_output.txt", "a") as f:
                    f.write(result)

    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C) to stop the script
        print("Script terminated by user.")

    finally:
        # Close the subprocess
        process.terminate()


if __name__ == "__main__":
    adb_shell_listener()
