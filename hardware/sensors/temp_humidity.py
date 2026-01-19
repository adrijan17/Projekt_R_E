import board
import adafruit_dht

# DHT11 na GPIO17
dht = adafruit_dht.DHT11(board.D17)

def read_temp_humidity():
  
    try:
        temp = dht.temperature
        hum = dht.humidity
        return round(temp, 2), round(hum, 2)
    except RuntimeError:
        
        return None, None
