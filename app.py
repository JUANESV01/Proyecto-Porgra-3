#By Juan

from fastapi import FastAPI, UploadFile, HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import uuid
import requests
import time

# Crear la aplicación FastAPI
aplicacion = FastAPI()

# Cargar el modelo de embeddings
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# Conectar a Qdrant (asume que estás usando localhost con Docker)
cliente_qdrant = QdrantClient(host="qdrant", port=6333)

# URL corregida para Ollama (endpoint generate)
OLLAMA_API_URL = "http://ollama:11434/api/generate"

# Nombre de la colección donde se guardarán los vectores
NOMBRE_COLECCION = "documentos_pdf"


def wait_for_services():
    """Esperar a que los servicios estén listos"""
    max_retries = 30

    # Esperar por Qdrant
    for i in range(max_retries):
        try:
            cliente_qdrant.get_collections()
            print("✅ Qdrant está listo")
            break
        except Exception:
            print(f"Esperando Qdrant... intento {i + 1}/{max_retries}")
            time.sleep(2)

    # Esperar por Ollama
    for i in range(max_retries):
        try:
            response = requests.get("http://ollama:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama está listo")
                break
        except Exception:
            print(f"Esperando Ollama... intento {i + 1}/{max_retries}")
            time.sleep(2)


# Inicializar servicios al arrancar
wait_for_services()

# Crear la colección si no existe
try:
    colecciones_existentes = cliente_qdrant.get_collections().collections
    if NOMBRE_COLECCION not in [c.name for c in colecciones_existentes]:
        cliente_qdrant.create_collection(
            collection_name=NOMBRE_COLECCION,
            vectors_config=VectorParams(
                size=384,  # Dimensión del modelo MiniLM
                distance=Distance.COSINE  # Métrica de similitud
            ),
        )
        print(f"✅ Colección '{NOMBRE_COLECCION}' creada")
    else:
        print(f"✅ Colección '{NOMBRE_COLECCION}' ya existe")
except Exception as e:
    print(f"Error al crear/verificar colección: {e}")


@aplicacion.post("/subir-pdf/")
async def subir_pdf(archivo: UploadFile):
    try:
        # Verificar que es un PDF
        if not archivo.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")

        # Leer el archivo PDF cargado
        lector = PdfReader(archivo.file)
        texto_total = "\n".join([pagina.extract_text() or "" for pagina in lector.pages])

        if not texto_total.strip():
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")

        # Dividir el texto en fragmentos (chunks) de 500 caracteres con overlap
        chunk_size = 500
        overlap = 50
        fragmentos = []

        for i in range(0, len(texto_total), chunk_size - overlap):
            fragmento = texto_total[i:i + chunk_size]
            if fragmento.strip():  # Solo agregar fragmentos no vacíos
                fragmentos.append(fragmento)

        # Obtener los vectores (embeddings) de cada fragmento
        vectores = modelo_embeddings.encode(fragmentos).tolist()

        # Crear los puntos con su ID, vector y texto original
        puntos = [
            PointStruct(
                id=str(uuid.uuid4()),  # Genera un ID único
                vector=vector,
                payload={
                    "texto": fragmento,
                    "archivo": archivo.filename,
                    "timestamp": int(time.time())
                }
            )
            for fragmento, vector in zip(fragmentos, vectores)
        ]

        # Insertar los vectores en Qdrant
        cliente_qdrant.upsert(collection_name=NOMBRE_COLECCION, points=puntos)

        return {
            "mensaje": f"{len(fragmentos)} fragmentos del archivo '{archivo.filename}' insertados correctamente.",
            "archivo": archivo.filename,
            "fragmentos_procesados": len(fragmentos)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el PDF: {str(e)}")


@aplicacion.post("/consultar/")
async def consultar_documentos(pregunta: str):
    try:
        if not pregunta.strip():
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

        # Convertir la pregunta a vector
        vector_pregunta = modelo_embeddings.encode(pregunta).tolist()

        # Buscar los fragmentos más similares en Qdrant
        resultados = cliente_qdrant.search(
            collection_name=NOMBRE_COLECCION,
            query_vector=vector_pregunta,
            limit=5,  # Obtener los 5 fragmentos más relevantes
            score_threshold=0.3  # Filtrar resultados con baja similitud
        )

        if not resultados:
            return {
                "pregunta": pregunta,
                "respuesta": "No se encontró información relevante en los documentos cargados.",
                "contexto": []
            }

        # Extraer el texto de los resultados
        fragmentos_encontrados = [r.payload["texto"] for r in resultados]

        # Crear el contexto combinado para Ollama
        contexto = "\n\n".join(fragmentos_encontrados)

        # Crear prompt para Ollama
        prompt = f"""Contexto de documentos:
{contexto}

Pregunta: {pregunta}

Instrucciones: Responde la pregunta basándote únicamente en el contexto proporcionado. Si la información no está disponible en el contexto, indica que no puedes responder con la información disponible.

Respuesta:"""

        # Enviar la consulta a Ollama con el endpoint correcto
        respuesta_ollama = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )

        if respuesta_ollama.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error al consultar Ollama: {respuesta_ollama.status_code}")

        respuesta_json = respuesta_ollama.json()
        texto_respuesta = respuesta_json.get("response", "").strip()

        return {
            "pregunta": pregunta,
            "respuesta": texto_respuesta,
            "contexto": fragmentos_encontrados,
            "num_fragmentos_encontrados": len(resultados)
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Timeout al consultar el modelo Ollama")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")


@aplicacion.get("/health/")
async def health_check():
    """Endpoint para verificar que todos los servicios estén funcionando"""
    health_status = {
        "api": "ok",
        "qdrant": "error",
        "ollama": "error",
        "timestamp": int(time.time())
    }

    # Verificar Qdrant
    try:
        collection_info = cliente_qdrant.get_collection(NOMBRE_COLECCION)
        health_status["qdrant"] = {
            "status": "ok",
            "points_count": collection_info.points_count
        }
    except Exception as e:
        health_status["qdrant"] = f"error: {str(e)}"

    # Verificar Ollama
    try:
        respuesta = requests.get("http://ollama:11434/api/tags", timeout=5)
        if respuesta.status_code == 200:
            models = respuesta.json().get('models', [])
            health_status["ollama"] = {
                "status": "ok",
                "models": [model['name'] for model in models]
            }
        else:
            health_status["ollama"] = f"error: status code {respuesta.status_code}"
    except Exception as e:
        health_status["ollama"] = f"error: {str(e)}"

    return health_status


@aplicacion.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "mensaje": "RAG API con Qdrant y Ollama",
        "version": "1.0",
        "endpoints": {
            "subir_pdf": "/subir-pdf/",
            "consultar": "/consultar/",
            "health": "/health/"
        }
    }
