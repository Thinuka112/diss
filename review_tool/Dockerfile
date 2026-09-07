# Build context = review_tool/  (the app imports review_store.py from this dir).
#   docker build -f review_tool/Dockerfile -t slice-review review_tool
FROM python:3.12-slim

WORKDIR /app/review_tool/webapp

# deps first (layer cache)
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# shared module + profiles live in the parent (review_tool/)
COPY review_store.py /app/review_tool/
COPY profiles /app/review_tool/profiles
# the web app itself (server, static, and the baked-in 500 review images)
COPY webapp /app/review_tool/webapp

# review_data (session logs + gold) is provided by a persistent VOLUME at runtime
ENV TARGET_REVIEWERS=1 \
    REVIEW_IMAGE_DIR=/app/review_tool/webapp/review_images \
    REVIEW_SESSIONS_DIR=/app/review_tool/review_data/sessions

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
