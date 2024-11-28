import asyncio
import websockets
import csv
import json
import logging

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Caminho do CSV
CSV_FILE_PATH = "plates_detected_jaime.csv"
WEBSOCKET_PORT = 8765

async def websocket_handler(websocket, path):
    """
    Manipulador para conexões WebSocket.
    """
    logging.info(f"Cliente conectado no caminho: {path}")
    try:
        await send_plate_numbers(websocket)
    except websockets.exceptions.ConnectionClosed:
        logging.info("Conexão encerrada pelo cliente.")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")

async def send_plate_numbers(websocket):
    """
    Envia os dados do CSV para o cliente conectado via WebSocket.
    """
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if "Plate_Number" not in reader.fieldnames:
                error_message = json.dumps({"error": "Column 'Plate_Number' not found in CSV"})
                await websocket.send(error_message)
                logging.error("Erro: Coluna 'Plate_Number' não encontrada no CSV.")
                return

            for row in reader:
                plate_number = row.get("Plate_Number")
                if plate_number:
                    # Criar JSON apenas com o campo Plate_Number
                    json_data = json.dumps({"plate_number": plate_number})
                    await websocket.send(json_data)  # Enviar para o cliente
                    logging.info(f"Enviado: {json_data}")
                    await asyncio.sleep(1)  # Simula envio em tempo real
                else:
                    logging.warning("Linha sem 'Plate_Number' encontrada.")
    except FileNotFoundError:
        error_message = json.dumps({"error": "CSV file not found"})
        await websocket.send(error_message)
        logging.error("Erro: Arquivo CSV não encontrado.")
    except Exception as e:
        logging.error(f"Erro ao enviar os dados: {e}")

async def start_websocket_server():
    """
    Inicializa o servidor WebSocket na porta especificada.
    """
    logging.info(f"Servidor WebSocket rodando na porta {WEBSOCKET_PORT}...")
    async with websockets.serve(websocket_handler, "0.0.0.0", WEBSOCKET_PORT):
        await asyncio.Future()  # Mantém o servidor ativo

if __name__ == "__main__":
    try:
        asyncio.run(start_websocket_server())
    except KeyboardInterrupt:
        logging.info("Servidor WebSocket encerrado manualmente.")
    except Exception as e:
        logging.error(f"Erro ao iniciar o servidor WebSocket: {e}")
