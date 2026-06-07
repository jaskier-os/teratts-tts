FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Model to pre-download and serve. Override at build time with --build-arg.
ARG TTS_MODEL=TeraTTS/glados2-g2p-vits
ENV TTS_MODEL=${TTS_MODEL}
ENV TTS_PORT=10018
ENV TTS_HOST=0.0.0.0

# Pre-download model and accentizer data at build time
RUN python -c "import os; from TeraTTS import TTS; TTS(os.environ['TTS_MODEL'], add_time_to_end=0.5, tokenizer_load_dict=True)" && \
    python -c "from ruaccent import RUAccent; a = RUAccent(); a.load(omograph_model_size='turbo', use_dictionary=True)"

EXPOSE 10018

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f "http://localhost:${TTS_PORT}/health" || exit 1

CMD ["python", "-u", "server.py"]
