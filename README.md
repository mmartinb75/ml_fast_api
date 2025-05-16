🧪 Cómo levantar el servidor FastAPI

  El servidor está definido en ml_server_2.py y expone endpoints para subir un modelo y hacer predicciones.

🔧 Requisitos

  Instala las dependencias necesarias:
  
  `pip install fastapi uvicorn scikit-learn pandas numpy`
  
🚀 Lanzar el servidor

  Desde la raíz del proyecto (donde esté ml_server_2.py):
  
  `fastapi dev ml_server_2.py`
  
  Esto abrirá el servidor en http://127.0.0.1:8000. La documentación interactiva está disponible en:
  http://127.0.0.1:8000/docs
  
📤 Endpoints principales

  POST /upload_model/: subir archivos .pkl del preprocesador, filtro y modelo.
  
  POST /predict/: realizar predicciones normales.
  
  POST /predict_w_intervals/: realizar predicciones con intervalos de confianza.
  
  Los tres archivos .pkl deben ser cargados antes de realizar predicciones.
