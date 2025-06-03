#By Juan

import requests
import time
import sys


def check_ollama_status():
    """Verificar si Ollama está en línea y tiene el modelo cargado"""
    ollama_url = "http://ollama:11434/api/tags"

    # Esperar a que Ollama esté disponible
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(ollama_url)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"Ollama está en línea. Modelos disponibles: {models}")

                # Verificar si el modelo deepseek-coder está disponible
                if any(model['name'] == 'deepseek-r1:1.5b' for model in models):
                    print("✅ El modelo deepseek-coder:1.5b-instruct está cargado y listo para usar.")
                    return True
                else:
                    print("❌ El modelo deepseek-coder:1.5b-instruct no está cargado todavía.")
            else:
                print(f"Ollama respondió con código de estado: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Intento {i + 1}/{max_retries}: Ollama no está disponible todavía. Error: {e}")

        time.sleep(10)  # Esperar 10 segundos antes de reintentar

    print("No se pudo conectar a Ollama después de varios intentos.")
    return False


if __name__ == "__main__":
    if check_ollama_status():
        sys.exit(0)  # Éxito
    else:
        sys.exit(1)  # Error