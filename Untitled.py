
from fastapi import FastAPI, UploadFile, HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import uuid
import requests
import json

# Crear la aplicación FastAPI
aplicacion = FastAPI()

# Cargar el modelo de embeddings
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# Conectar a Qdrant (asume que estás usando localhost con Docker)
cliente_qdrant = QdrantClient(host="qdrant", port=6333)

# URL de la API de Ollama
OLLAMA_API_URL = "http://ollama:11434/api/generate"

# Nombre de la colección donde se guardarán los vectores
NOMBRE_COLECCION = "documentos_pdf"

# Crear la colección si no existe
colecciones_existentes = cliente_qdrant.get_collections().collections
if NOMBRE_COLECCION not in [c.name for c in colecciones_existentes]:
    cliente_qdrant.create_collection(
        collection_name=NOMBRE_COLECCION,
        vectors_config=VectorParams(
            size=384,  # Dimensión del modelo MiniLM
            distance=Distance.COSINE  # Métrica de similitud
        ),
    )

@aplicacion.post("/subir-pdf/")
async def subir_pdf(archivo: UploadFile):
    # Leer el archivo PDF cargado
    lector = PdfReader(archivo.file)
    texto_total = "\n".join([pagina.extract_text() or "" for pagina in lector.pages])

    # Dividir el texto en fragmentos (chunks) de 500 caracteres
    fragmentos = [texto_total[i:i+500] for i in range(0, len(texto_total), 500)]

    # Obtener los vectores (embeddings) de cada fragmento
    vectores = modelo_embeddings.encode(fragmentos).tolist()

    # Crear los puntos con su ID, vector y texto original
    puntos = [
        PointStruct(
            id=str(uuid.uuid4()),  # Genera un ID único
            vector=vector,
            payload={"texto": fragmento}
        )
        for fragmento, vector in zip(fragmentos, vectores)
    ]

    # Insertar los vectores en Qdrant
    cliente_qdrant.upsert(collection_name=NOMBRE_COLECCION, points=puntos)

    return {"mensaje": f"{len(fragmentos)} fragmentos insertados en Qdrant correctamente."}

@aplicacion.post("/consultar/")
async def consultar_documentos(pregunta: str):
    try:
        # Convertir la pregunta a vector
        vector_pregunta = modelo_embeddings.encode(pregunta).tolist()
        
        # Buscar los fragmentos más similares en Qdrant
        resultados = cliente_qdrant.search(
            collection_name=NOMBRE_COLECCION,
            query_vector=vector_pregunta,
            limit=5  # Obtener los 5 fragmentos más relevantes
        )
        
        # Extraer el texto de los resultados
        fragmentos_encontrados = [r.payload["texto"] for r in resultados]
        
        # Crear el contexto combinado para Ollama
        contexto = "\n\n".join(fragmentos_encontrados)
        
        # Crear prompt para Ollama
        prompt = f"""Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""
        
        # Enviar la consulta a Ollama
        respuesta_ollama = requests.post(
            OLLAMA_API_URL,
            json={
                "model": "deepseek-r1-5b",
                "prompt": prompt,
                "stream": False
            }
        )
        
        if respuesta_ollama.status_code != 200:
            raise HTTPException(status_code=500, detail="Error al consultar el modelo Ollama")
        
        # Extraer la respuesta generada
        texto_respuesta = respuesta_ollama.json().get("response", "")
        
        return {
            "pregunta": pregunta,
            "respuesta": texto_respuesta,
            "contexto": fragmentos_encontrados
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {str(e)}")

@aplicacion.get("/health/")
async def health_check():
    """Endpoint para verificar que todos los servicios estén funcionando"""
    health_status = {
        "api": "ok",
        "qdrant": "error",
        "ollama": "error"
    }
    
    # Verificar Qdrant
    try:
        colecciones = cliente_qdrant.get_collections()
        health_status["qdrant"] = "ok"
    except Exception as e:
        health_status["qdrant"] = f"error: {str(e)}"
    
    # Verificar Ollama
    try:
        respuesta = requests.get("http://ollama:11434/api/tags")
        if respuesta.status_code == 200:
            health_status["ollama"] = "ok"
        else:
            health_status["ollama"] = f"error: status code {respuesta.status_code}"
    except Exception as e:
        health_status["ollama"] = f"error: {str(e)}"
    
    return health_status