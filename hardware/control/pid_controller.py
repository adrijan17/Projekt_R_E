from simple_pid import PID
from config.config import KP, KI, KD, SOIL_SETPOINT, PID_MIN, PID_MAX

def create_pid():
    pid = PID(KP, KI, KD, setpoint=SOIL_SETPOINT)
    pid.output_limits = (PID_MIN, PID_MAX)
    return pid
