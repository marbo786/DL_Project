FROM python:3.12-slim
WORKDIR /app
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY api /app/api
COPY web/public /app/web/public
COPY release/model.npz release/summary.json release/ablations.json release/calibration.json release/robustness.json /app/release/
RUN adduser --system --uid 10001 forgerylens
USER 10001
ENV PORT=8000 OPENBLAS_NUM_THREADS=1
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port \"$PORT\" --no-access-log"]