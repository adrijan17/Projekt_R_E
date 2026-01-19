import time
import board
import adafruit_dht

# DHT11 na GPIO 17
dht = adafruit_dht.DHT11(board.D17)

print("Pokrenut DHT11 test... CTRL+C za izlaz")

try:
    while True:
        try:
            temperature = dht.temperature
            humidity = dht.humidity

            if temperature is not None and humidity is not None:
                print(f"Temperatura: {temperature} °C")
                print(f"Vlaga zraka: {humidity} %")
                print("-" * 30)
            else:
                print("Neuspjelo očitanje")

        except RuntimeError as e:
            print("Greška:", e)

        time.sleep(2)

except KeyboardInterrupt:
    print("\nPrekid programa")

finally:
    dht.exit()