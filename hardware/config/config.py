# PID kontroler parametri
KP = 1.2                # proporcionalni
KI = 0.05               # integralni
KD = 0.1                # derivativni

SOIL_SETPOINT = 70.0    # željena vlaga tla (%)
PID_MIN = 0             # minimalni izlaz PID-a (%)
PID_MAX = 100           # maksimalni izlaz PID-a (%)

# Simulacija tla parametri
EVAPORATION_RATE = 0.2   # bazna evaporacija po ciklusu
PUMP_EFFECT = 0.5        # efekt pumpe (koliko % vlage dodaje)
PUMP_FLOW_RATE = 10      # ml/s, sporija prskalica
SOIL_CAPACITY = 1000     # ml, veličina kante

# loop
SAMPLE_TIME = 1.0        # sekunde između ciklusa

# data logging
DATA_FOLDER = "data"                   # folder za logove
DATA_FILE = f"{DATA_FOLDER}/logs.csv"  # CSV fajl

#pinovi
LED_PIN = 18       # GPIO za pumpu / LED
LDR_PIN = 27          # GPIO za svjetlosni senzor
DHT_PIN = 17          # GPIO za DHT11
