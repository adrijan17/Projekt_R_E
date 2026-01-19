import time
import csv
from datetime import datetime
import RPi.GPIO as GPIO
import os

from config.config import *
from sensors.temp_humidity import read_temp_humidity
from sensors.ldr_sensor import read_light
from sensors.soil_simulator import SoilSimulator
from control.pid_controller import create_pid


os.makedirs(DATA_FOLDER, exist_ok=True)

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

soil = SoilSimulator(initial_moisture=40.0, soil_capacity=SOIL_CAPACITY)
pid = create_pid()

# kreira csv
with open(DATA_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time",
        "temperature",
        "air_humidity",
        "soil_moisture",
        "light",
        "pump_output"
    ])

try:
    while True:
        # očitavanje senzora
        temperature, air_humidity = read_temp_humidity()
        light = read_light()

        if temperature is None or air_humidity is None:
            print("Neuspjelo očitanje DHT11")
            time.sleep(SAMPLE_TIME)
            continue

        # pid kontrola
        pump_output = pid(soil.moisture)

        # simulacija tla
        pump_on = pump_output > 0  # ledica uključena ako PID > 0
        soil_moisture = soil.update(
            pump_on=pump_on,
            pump_flow_rate=PUMP_FLOW_RATE,
            temperature=temperature,
            dt=SAMPLE_TIME
        )

        # upravljanje pumpom/ledicom
        if pump_on:
            GPIO.output(LED_PIN, GPIO.HIGH)
            pump_status = "ON"
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
            pump_status = "OFF"

        # pisanje podataka u CSV
        now = datetime.now().strftime("%H:%M:%S")
        with open(DATA_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now,
                temperature,
                air_humidity,
                soil_moisture,
                light,
                round(pump_output, 2)
            ])

        
        print(f"[{now}] "
              f"T={temperature}°C RH={air_humidity}% "
              f"Soil={soil_moisture}% "
              f"Light={'DOVOLJNO' if not light else 'SLABO'} "
              f"Pump={pump_status} ({round(pump_output,2)}%)")

        time.sleep(SAMPLE_TIME)

except KeyboardInterrupt:
    print("\nSustav zaustavljen.")

finally:
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
