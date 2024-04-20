FROM python:3.10
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libhdf5-serial-dev
RUN pip install -r requirements.txt
EXPOSE 5050
ENTRYPOINT ["gunicorn", "app:app"]





