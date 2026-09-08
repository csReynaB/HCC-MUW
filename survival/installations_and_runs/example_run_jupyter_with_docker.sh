docker run --rm -it \
  --name phipsurv-jupyter \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e "JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime-$(id -u)" \
  -e "MPLCONFIGDIR=/tmp/matplotlib-$(id -u)" \
  -p 8888:8888 \
  -v "$PWD:/workspace" \
  phipsurv:latest \
  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --ServerApp.root_dir=/workspace
