import RPi.GPIO as GPIO

LDR_PIN = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(LDR_PIN, GPIO.IN)

def read_light():
    # true DOVOLJNO SVJETLO, false SLABO SVJETLO
    return GPIO.input(LDR_PIN)
