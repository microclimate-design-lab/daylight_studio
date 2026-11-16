# Daylight Studio – container for free hosts (Render, Koyeb, ...) or Hugging Face Spaces.
#   docker build -t daylight-studio .
#   docker run -p 7860:7860 daylight-studio      ->  http://localhost:7860
# Runs without torch (NumPy backend) to fit in 512 MB of memory.
FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DAYLIGHT_BACKEND=numpy \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user -r requirements-space.txt

COPY --chown=user . .

EXPOSE 7860
# one worker (models are loaded once), a few threads for concurrent requests; the host may set PORT
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120
