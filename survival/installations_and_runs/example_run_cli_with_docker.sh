docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/workspace" \
  -v /home/creyna/Vogl-lab_Projects_git/HCC:/home/creyna/Vogl-lab_Projects_git/HCC:ro \
  -w /workspace \
  phipsurv:latest \
  phipsurv -c /workspace/configs/config_survival_HCC.yaml
