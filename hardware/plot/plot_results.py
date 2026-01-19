import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import time

import sys, os
from config.config import DATA_FILE


SOIL_FILE = "soil_pump.png"
TEMP_FILE = "temp_humidity.png"
LDR_FILE = "ldr.png"

REFRESH_INTERVAL = 1.0

def plot_soil(df):
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df['time'], df['soil_moisture'], 'g-', label='Soil Moisture (%)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Soil Moisture (%)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()
    ax2.plot(df['time'], df['pump_output'], 'r-', alpha=0.7, label='Pump Output (%)')
    ax2.set_ylabel('Pump Output (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle('Soil Moisture and Pump Output')
    fig.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(SOIL_FILE)
    plt.close(fig)

def plot_temp_humidity(df):
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df['time'], df['temperature'], 'b-', label='Temperature (°C)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(df['time'], df['air_humidity'], 'c-', alpha=0.7, label='Air Humidity (%)')
    ax2.set_ylabel('Air Humidity (%)', color='c')
    ax2.tick_params(axis='y', labelcolor='c')

    fig.suptitle('Temperature and Air Humidity')
    fig.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(TEMP_FILE)
    plt.close(fig)

def plot_ldr(df):
    plt.figure(figsize=(12,3))
    plt.plot(df['time'], df['light'], 'orange', label='LDR Light')
    plt.xlabel('Time')
    plt.ylabel('Light (0/1)')
    plt.title('LDR Sensor')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LDR_FILE)
    plt.close()

def main_loop():
    print("Starting headless live plotting... CTRL+C to stop")
    while True:
        try:
            df = pd.read_csv(DATA_FILE)
            if df.empty:
                time.sleep(REFRESH_INTERVAL)
                continue

            df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')

            # crtanje grafova
            plot_soil(df)
            plot_temp_humidity(df)
            plot_ldr(df)

            time.sleep(REFRESH_INTERVAL)
        except KeyboardInterrupt:
            print("Stopping live plotting.")
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main_loop()
