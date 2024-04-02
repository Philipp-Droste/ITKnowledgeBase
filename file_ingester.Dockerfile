FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY file_ingester.py .
COPY chains.py .
COPY stream_handler.py .
COPY job_posts.csv .

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "file_ingester.py", "--server.port=8503", "--server.address=0.0.0.0"]