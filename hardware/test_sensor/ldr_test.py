import RPi.GPIO as GPIO
import time

# ----- POSTAVKE -----
LDR_PIN = 27  # GPIO pin gdje je LDR
SAMPLE_TIME = 1  # sekunde između očitanja

# ----- INICIJALIZACIJA GPIO -----
GPIO.setmode(GPIO.BCM)
GPIO.setup(LDR_PIN, GPIO.IN)

# ----- FUNKCIJA ZA ČITANJE -----
def read_light():
    """
    Vraća True ako je svjetlo dovoljno (HIGH),
    False ako je tamno (LOW)
    """
    return GPIO.input(LDR_PIN)

# ----- GLAVNA PETLJA -----
print("Pokrenut LDR test na GPIO 27... CTRL+C za izlaz")

try:
    while True:
        light = read_light()
        if light:
            print("Svjetlo: DOVOLJNO / HIGH")
        else:
            print("Svjetlo: SLABO / LOW")
        time.sleep(SAMPLE_TIME)

except KeyboardInterrupt:
    print("\nPrekid programa")

finally:
    GPIO.cleanup()
    print("GPIO očišćen, kraj programa.")
