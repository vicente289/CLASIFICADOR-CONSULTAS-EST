# Clasificador de Consultas Estudiantiles

Proyecto academico de Inteligencia Artificial para clasificar consultas estudiantiles.

## Ejecucion rapida

Desde esta carpeta raiz:

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 train_model.py
py -3.10 -m streamlit run app.py
```

Tambien puedes usar:

```powershell
.\run_app.bat
```

La aplicacion principal esta en:

```text
clasificador_consultas_estudiantiles/app.py
```

## Publicar para que cualquier persona la vea

La app esta preparada para Streamlit Community Cloud.

Configuracion recomendada:

- Repositorio: `vicente289/CLASIFICADOR-CONSULTAS-EST`
- Rama: `main`
- Archivo principal: `app.py`
- Dependencias: `requirements.txt`
- Tema visual: `.streamlit/config.toml`

Pasos completos en:

```text
DEPLOYMENT.md
```
