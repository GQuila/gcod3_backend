# GCOD3 - quick local dev launcher (one command)
# Usage:  bash start.sh
set -e
echo ">> Installing backend dependencies"
pip install -r requirements.txt
echo ">> Starting FastAPI (http://localhost:8001)"
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
