import subprocess
import sys
INTERPRETER = "python3"

powerLogger = subprocess.Popen([INTERPRETER, "powerLogger/powerLogger.py"], shell=True, stdout=sys.stdout, stderr=sys.stderr, stdin=None, text=True)
adbParser = subprocess.Popen([INTERPRETER, "adbParser.py"], shell=True, stdout=sys.stdout, stderr=sys.stderr, stdin=None, text=True)

adbParser.wait()
powerLogger.kill()
