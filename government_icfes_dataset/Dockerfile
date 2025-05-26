FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY despliegue/app.py .
COPY convolutional.py .
COPY despliegue/datosylayout.py .
COPY despliegue/prediccion.py .
COPY despliegue/tab_estu_mcpio_reside.py .
COPY label_encoder_unk.py .

COPY data/ ./data/

COPY best_models/ ./best_models/

COPY encoders/ ./encoders/

EXPOSE 8050

CMD ["python", "app.py"]