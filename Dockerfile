FROM python:3.11-slim
WORKDIR /pvenergy

COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgomp1 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

COPY data/raw/pvdaq/metadata.csv ./data/raw/pvdaq/metadata.csv
COPY data/raw/pvdaq/metric_ids.csv ./data/raw/pvdaq/metric_ids.csv
COPY data/merged ./data/merged
COPY data/results ./data/results
COPY data/models ./data/models

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=pvenergy.settings
ENV PYTHONPATH="/pvenergy/src"

RUN pvenergy django migrate

ENTRYPOINT ["pvenergy"]
CMD ["runserver"]