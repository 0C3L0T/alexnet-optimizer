import serial
import time

def arduino_listener():
    # get arduino serial port
    try: 
        arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)
    except:
        arduino = serial.Serial(port='/dev/ttyACM1', baudrate=9600, timeout=.1)

    try:
        while True: 
            data = arduino.readline()[:-2] #the last bit gets rid of the new-line chars
            data = data.decode('utf-8')
            data = data.split(',')

            # first output only has two values
            if len(data) != 3:
                continue
    
            [curr_V, curr_mA, power_mW] = data

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            result = f"[{timestamp}] Voltage: {curr_V}V, Current: {curr_mA}mA, Power: {power_mW}mW\n"
            print(result)

            with open("power_output.txt", "a") as f:
                f.write(result)
        
    except KeyboardInterrupt:
        print("Script terminated by user.")

    finally:
        arduino.close()

if __name__ == "__main__":
    arduino_listener()