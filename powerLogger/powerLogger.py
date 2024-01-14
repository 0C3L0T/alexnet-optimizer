import serial 

try: 
    arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)
except:
    arduino = serial.Serial(port='/dev/ttyACM1', baudrate=9600, timeout=.1)


while True: 
    data = arduino.readline()[:-2] #the last bit gets rid of the new-line chars
    data = data.decode('utf-8')
    data = data.split(',')

    if len(data) != 3:
        continue
    
    [curr_V, curr_mA, power_mW] = data

    print(f"Voltage: {curr_V}V, Current: {curr_mA}mA, Power: {power_mW}mW (from Python)")


