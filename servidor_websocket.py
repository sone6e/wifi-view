#!/usr/bin/env python3
"""
Servidor WebSocket que escucha UDP del ESP32-S3 y lo expone como WebSocket
para que SensingClient pueda conectarse.

Uso:
    python servidor_websocket.py

Luego en otra terminal:
    python radar.py
"""

import asyncio
import socket
import websockets
from threading import Thread
from queue import Queue

# Cola para pasar datos de UDP a WebSocket
data_queue = Queue()


def udp_listener():
    """Escucha UDP en puerto 5005 (donde envía el ESP32-S3)"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 5005))
    print("📡 Servidor UDP escuchando en puerto 5005...")
    print("   (Asegúrate de que tu ESP32-S3 envía a este puerto)\n")
    print("⏳ Esperando datos UDP...\n")
    
    try:
        count = 0
        while True:
            data, addr = sock.recvfrom(4096)
            count += 1
            print(f"✅ Paquete #{count}: {len(data)} bytes de {addr[0]}:{addr[1]}")
            data_queue.put(data)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


async def handle_websocket(ws):
    """Maneja conexiones WebSocket de clientes"""
    print(f"🔗 Cliente WebSocket conectado desde {ws.remote_address}")
    try:
        while True:
            # Intenta obtener datos de la cola con timeout
            try:
                data = data_queue.get(timeout=1)
                await ws.send(data)
            except:
                # Timeout en la cola, solo esperamos el siguiente
                await asyncio.sleep(0.1)
    except websockets.exceptions.ConnectionClosed:
        print(f"❌ Cliente WebSocket desconectado")
    except Exception as e:
        print(f"⚠️  Error: {e}")


async def start_websocket_server():
    """Inicia servidor WebSocket en puerto 8000"""
    print("🌐 Iniciando servidor WebSocket en ws://127.0.0.1:8000\n")
    
    server = await websockets.serve(handle_websocket, "127.0.0.1", 8000)
    print("✅ Servidor WebSocket listo\n")
    print("=" * 60)
    print("Próximos pasos:")
    print("1. Abre otra terminal")
    print("2. Ejecuta: python radar.py")
    print("=" * 60 + "\n")
    
    await asyncio.Future()  # run forever


def main():
    print("\n" + "=" * 60)
    print("🚀 SERVIDOR UDP → WebSocket para RuView CSI")
    print("=" * 60 + "\n")
    
    # Inicia listener UDP en background
    udp_thread = Thread(target=udp_listener, daemon=True)
    udp_thread.start()
    
    # Inicia servidor WebSocket
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(start_websocket_server())
    except KeyboardInterrupt:
        print("\n\n⏹️  Servidor detenido")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
