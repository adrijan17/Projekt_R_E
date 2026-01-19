import time
import RPi.GPIO as GPIO
import board
import adafruit_dht

# ----- PINOVI -----
BUZZER_PIN = 18   # GPIO za buzzer
LDR_PIN = 27      # GPIO za LDR
DHT_PIN = board.D17  # DHT11 na GPIO 17

SAMPLE_TIME = 2  # sekunde

# ----- INICIJALIZACIJA GPIO -----
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LDR_PIN, GPIO.IN)

# ----- INICIJALIZACIJA DHT11 -----
dht = adafruit_dht.DHT11(DHT_PIN)

print("Pokrenut test DHT11 + LDR + buzzer... CTRL+C za izlaz")

try:
    while True:
        # --- DHT11 ---
        try:
            temperature = dht.temperature
            humidity = dht.humidity
        except RuntimeError as e:
            print("Greška DHT11:", e)
            temperature = None
            humidity = None

        # --- LDR ---
        light = GPIO.input(LDR_PIN)

        # --- Buzzer ---
        if temperature is not None:
            if temperature < 30:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                buzzer_status = "ON"
            else:
                GPIO.output(BUZZER_PIN, GPIO.LOW)
                buzzer_status = "OFF"
        else:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            buzzer_status = "OFF"

        # --- Ispis ---
        print("-" * 30)
        if temperature is not None and humidity is not None:
            print(f"Temperatura: {temperature} °C  Vlaga: {humidity}%")
        else:
            print("Neuspjelo očitanje DHT11")
        print(f"LDR (svjetlo): {'DOVOLJNO' if light else 'SLABO'}")
        print(f"Buzzer: {buzzer_status}")
        print("-" * 30)

        time.sleep(SAMPLE_TIME)

except KeyboardInterrupt:
    print("\nPrekid programa")

finally:
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()
    dht.exit()
    print("GPIO očišćen, kraj programa.")
