#include <Adafruit_INA219.h>

//declare timer trigger flag and counter value
volatile boolean triggered = false;

//declare INA219 variables
Adafruit_INA219 ina219;
float current_mA = 0.0;
float loadVoltage = 0.0;
float power_mW = 0.0;
// float energy_mWh = 0.0;
unsigned long elapsed = 0;

void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);

  //setup the INA219
  ina219.begin();

  // stop interrupts
  cli();

  // TIMER 1 for interrupt frequency 1 Hz:

  //initialise the CCR register and the counter
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  
  // set compare match register for 10 Hz increments
  OCR1A = 12499; // = 8000000 / (64 * 10) - 1 (must be <65536)
  
  // turn on CTC mode
  TCCR1B |= (1 << WGM12);
  
  // Set CS12, CS11 and CS10 bits for 64 prescaler
  TCCR1B |= (0 << CS12) | (1 << CS11) | (1 << CS10);
  
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);

  // allow interrupts
  sei();
}

void loop() {
  if (triggered) {
    //get the values measured by the INA219
    ina219values();

    Serial.println(
      String(loadVoltage) + ","
      + String(current_mA)+ ","
      + String(power_mW)
      // + String(energy_mWh)
      );

    //reset the flag
    triggered = false;
  }
}

// attach timer compare interrupt?
ISR(TIMER1_COMPA_vect){
  triggered = true;
}

void ina219values() {
  float shuntVoltage = 0.0;
  float busVoltage = 0.0;
  
  //turn the INA219 on
  ina219.powerSave(false);
  
  //get the shunt voltage, bus voltage, current and power consumed from the INA219
  shuntVoltage = ina219.getShuntVoltage_mV();
  busVoltage = ina219.getBusVoltage_V();
  current_mA = ina219.getCurrent_mA();
  elapsed = millis();

  //turn the INA219 off
  ina219.powerSave(true);

  //compute the load voltage
  loadVoltage = busVoltage + (shuntVoltage / 1000.0);

  //compute the power consumed
  power_mW = loadVoltage*current_mA;
  
  //compute the energy consumed (t = elapsed[ms] / 3600[s/h] * 1000[ms/s])
  // energy_mWh += power_mW * ( elapsed / 3600000.0);
}
