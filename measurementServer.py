import subprocess
INTERPRETER = "python3"

powerLogger = subprocess.Popen([INTERPRETER, "powerLogger/powerLogger.py"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
adbParser = subprocess.Popen([INTERPRETER, "adbParser.py"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
