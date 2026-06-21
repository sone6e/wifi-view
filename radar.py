import asyncio
import websockets
import json

async def radar_definitivo():
    print("🚀 Apuntando el radar a las rutas del código fuente...\n")
    
    # Las dos verdades absolutas que encontramos en stream.py
    rutas_reales = [
        "ws://127.0.0.1:8000/api/v1/stream/pose",
        "ws://127.0.0.1:8000/api/v1/stream/events"
    ]
    
    # El salvoconducto para FastAPI
    cabeceras = {"Origin": "http://127.0.0.1:8000"}

    for uri in rutas_reales:
        print(f"🔗 Conectando a: {uri} ...")
        try:
            async with websockets.connect(uri, additional_headers=cabeceras) as ws:
                print(f"\n🎉 ¡VICTORIA! El túnel se abrió perfectamente en {uri}")
                print("📡 Atrayendo la Matrix en vivo (presiona Ctrl+C para salir):\n" + "="*60)
                
                while True:
                    mensaje = await ws.recv()
                    datos = json.loads(mensaje)
                    print(f"🕺 {datos}")
                    
        except Exception as e:
            print(f"❌ Rechazado: {e}\n")

if __name__ == "__main__":
    try:
        asyncio.run(radar_definitivo())
    except KeyboardInterrupt:
        print("\n⏹️ Radar cerrado.")