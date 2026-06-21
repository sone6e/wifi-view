#!/usr/bin/env python3
"""
Script para monitorear tráfico UDP y encontrar de dónde viene
"""

import socket
import sys

print("=" * 70)
print("🔍 MONITOR UDP - Escuchando TODOS los puertos UDP")
print("=" * 70)
print()
print("Este script escucha en puerto 5005")
print("Si el ESP32 envía a otro puerto, lo veremos aquí")
print()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Intenta múltiples puertos
puertos_a_probar = [5005, 5006, 5555, 8888, 9999, 12345]

for puerto in puertos_a_probar:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", puerto))
        print(f"✅ Escuchando en puerto {puerto}...")
        break
    except:
        print(f"❌ Puerto {puerto} ocupado, probando el siguiente...")

print()
print("⏳ Esperando datos... (Presiona Ctrl+C para salir)")
print()

try:
    count = 0
    while True:
        data, addr = sock.recvfrom(4096)
        count += 1
        print(f"📨 Paquete #{count} recibido de {addr[0]}:{addr[1]} -> {len(data)} bytes")
        # Muestra primeros 50 bytes
        print(f"   Primeros 50 bytes: {data[:50]}")
        print()
except KeyboardInterrupt:
    print(f"\n\n✋ Monitoreo detenido. Total: {count} paquetes recibidos")
finally:
    sock.close()
