FROM python:3.14-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 9000

CMD ["uvicorn", "api.entry:create_api", "--host", "0.0.0.0", "--port", "9000", "--reload"]
