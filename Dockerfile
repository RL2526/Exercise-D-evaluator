 # Dockerfile for exercise-d-evaluator
# Builds a container that:
#   - contains your env, opponents, and evaluator code
#   - expects:
#       /submission  -> mounted student repo (read-only)
#       /out         -> mounted output dir for result.json
#   - runs:
#       python -m evaluator.run --submission /submission --out /out/result.json

FROM python:3.10-slim

# 1) System setup (optional but usually useful for RL / scientific Python)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

# 2) Workdir inside the container
WORKDIR /app

# 3) Copy dependency file(s) first so Docker caching works well
# If you use requirements.txt:
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# If you use pyproject.toml/poetry instead, comment the two lines above
# and replace with your preferred installer.

# 4) Copy the rest of the evaluator code
# Make sure your repo has a package like `evaluator/` with __init__.py and run.py
COPY . .

# 5) Environment settings (nice-to-have)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 6) Default command:
# This is what your GitHub Action will run in docker:
#   docker run -v "$PWD:/submission:ro" -v "$PWD/out:/out" exd-eval
CMD ["python", "-m", "evaluator.evaluation", "--submission", "/submission", "--out", "/out/result.json"]
