FROM python:3.11-slim

WORKDIR /pvenergy

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./

RUN pip install --no-cache-dir .

COPY src/ ./src/

RUN pip install --no-cache-dir .

COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=pvenergy.settings
ENV PYTHONPATH="/pvenergy/src"

RUN pvenergy django migrate

ENTRYPOINT ["pvenergy"]
CMD ["runserver"]