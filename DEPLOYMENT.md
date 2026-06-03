# Publicar la app

Este proyecto ya esta preparado para publicarse como una app web de Streamlit.

## Opcion recomendada: Streamlit Community Cloud

1. Sube los cambios a GitHub.
2. Entra a https://share.streamlit.io.
3. Selecciona el repositorio:
   `vicente289/CLASIFICADOR-CONSULTAS-EST`
4. Selecciona la rama:
   `main`
5. Archivo principal:
   `app.py`
6. En configuracion avanzada, selecciona Python 3.10 si la plataforma lo permite.
7. Haz clic en Deploy.

La plataforma instalara las dependencias desde `requirements.txt` y usara el tema definido en `.streamlit/config.toml`.

## Opcion alternativa: Render

El repositorio incluye `render.yaml` y `Procfile`.

1. Crea una cuenta en https://render.com.
2. Crea un nuevo Web Service desde el repositorio de GitHub.
3. Usa:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --browser.gatherUsageStats=false`

## Nota sobre retroalimentacion

La app permite guardar retroalimentacion en `feedback_consultas.csv`. En un despliegue gratuito, ese archivo puede no ser permanente si el servicio reinicia. Para una version publica final, conviene conectar esta retroalimentacion a una base de datos o desactivar el guardado si se quiere evitar almacenar consultas reales.

## Comprobacion local antes de publicar

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 train_model.py
py -3.10 -m pytest
py -3.10 -m streamlit run app.py
```
