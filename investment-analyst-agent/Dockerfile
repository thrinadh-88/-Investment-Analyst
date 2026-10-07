FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/       ./app/
COPY dashboard/ ./dashboard/
COPY reports/   ./reports/

# Streamlit config — disable telemetry and set headless mode
RUN mkdir -p /app/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\n\n[browser]\ngatherUsageStats = false\n' \
    > /app/.streamlit/config.toml

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
