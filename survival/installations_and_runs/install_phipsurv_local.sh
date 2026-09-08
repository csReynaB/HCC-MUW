# Get python env with necessary libs first
# Check ML.yml file to see what needs to be installed
# Micromamba way:
micromamba create \
  --yes \
  --name phipsurv \
  --file ML_env.yml

micromamba activate phipsurv

# once all packages are installed and env activated, run this in the head repo folder:
python -m pip install \
  --editable . \
  --no-build-isolation \
  --no-deps

# check everyhing works:
python -c "import phipsurv; print(phipsurv.__file__)"
phipsurv -h
