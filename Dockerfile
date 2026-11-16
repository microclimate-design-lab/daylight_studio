# Daylight Studio – container for Hugging Face Spaces (Docker SDK) or any Docker host.
#   docker build -t daylight-studio .
#   docker run -p 7860:7860 daylight-studio      ->  http://localhost:7860
FROM python:3.13-slim

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user -r requirements-space.txt

COPY --chown=user . .

EXPOSE 7860
# one worker (models are loaded once, ~1 GB RAM), threads for concurrent requests
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "120"]
