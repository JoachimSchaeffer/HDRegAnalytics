export PYTHONPATH=/home/docker-user/:$PYTHONPATH

JUPYTER_PASSWORD='HDfeatlin'
JUPYTER_PORT=8056
jupyter-lab --allow-root --no-browser --port 8056 --ip 0.0.0.0