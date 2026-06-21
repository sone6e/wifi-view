import serial
import time

# Reemplaza COM7 con el puerto que encontraste
puerto = "COM5"
velocidad = 115200  # Velocidad típica del ESP32

try:
    ser = serial.Serial(puerto, velocidad, timeout=1)
    print(f"📡 Conectado a {puerto} a {velocidad} baud")
    print("⏳ Leyendo datos...\n")
    
    while True:
        if ser.in_waiting:
            linea = ser.readline().decode('utf-8', errors='ignore')
            print(linea, end='')
except KeyboardInterrupt:
    print("\n⏹️  Detenido")
finally:
    if ser.is_open:
        ser.close()