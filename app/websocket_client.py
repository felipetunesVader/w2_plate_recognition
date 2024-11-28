import asyncio
import websockets

# URL do WebSocket
WEBSOCKET_URL = "ws://localhost:8765"

async def test_client():
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            print("Conectado ao servidor WebSocket.")
            while True:
                message = await websocket.recv()
                print(f"Mensagem recebida: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Conexão encerrada pelo servidor.")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    asyncio.run(test_client())
