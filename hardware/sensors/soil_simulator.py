import random

class SoilSimulator:
    
    # Simulacija vlage tla u kantici (1000 ml) sa prskalicom
    
    def __init__(self, initial_moisture=40.0, soil_capacity=1000):
        self.moisture = initial_moisture      
        self.soil_capacity = soil_capacity    

    def update(self, pump_on=False, pump_flow_rate=10, 
               temperature=25, dt=1.0):
       

        # efekt pumpanja
        if pump_on:
            
            added_water = pump_flow_rate * dt       # ml
            added_percent = (added_water / self.soil_capacity) * 100
            self.moisture += added_percent

        # protok 
        absorption_loss = random.uniform(0.0, 0.2)
        self.moisture -= absorption_loss

        # ovisnost o temperaturi
        evap_rate = 0.05 + (temperature - 20) * 0.01
        self.moisture -= evap_rate * dt

        # ograničenje vlage između 0 i 100%
        self.moisture = max(0.0, min(100.0, self.moisture))

        return round(self.moisture, 2)
