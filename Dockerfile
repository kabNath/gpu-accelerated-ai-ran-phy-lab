FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .
CMD ["bash", "-lc", "pytest -v && python benchmarks/run_end_to_end.py"]
