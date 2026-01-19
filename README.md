# Projekt Digitalna, inteligentna i prediktivna poljoprivreda

## Hardware Setup

This project uses the following components connected to the Raspberry Pi:

| Component                        | GPIO / Connection | Notes                                    |
|----------------------------------|-----------------|--------------------------------------------|
| **DHT11 Temperature & Humidity** | GPIO 17         | Reads air temperature and humidity         |
| **LDR (Light Sensor)**           | GPIO 27         | Measures light level (0–1)                 |
| **LED lamp (Pump Indicator)**    | GPIO 18         | Turns on when irrigation is needed         |
| **Power**                        | 5V / GND        | Powers sensors and buzzer from the Pi      |

### Wiring Notes
- **DHT11**: VCC → 3.3V, GND → GND, DATA → GPIO17  
- **LDR**: Connected as a voltage divider, output → GPIO27  
- **LED lamp**: GPIO18 → Positive, GND → Ground  

### Optional
- Soil moisture is **simulated in software**; a real soil sensor can replace it.  
- If using a **water pump**, connect it via a relay to protect the Pi’s GPIO.
