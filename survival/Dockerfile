FROM mambaorg/micromamba:2.8.1

LABEL org.opencontainers.image.title="phipsurv" \
      org.opencontainers.image.description="Survival modelling and plotting for PhIP-seq data"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /opt/phipsurv

# Install into the image's base environment. The micromamba entrypoint
# automatically activates this environment for both CLI and Jupyter commands.
COPY --chown=$MAMBA_USER:$MAMBA_USER ML_env.yml /tmp/ML_env.yml
RUN micromamba install --yes --name base --file /tmp/ML_env.yml \
    && micromamba clean --all --yes

# Activate the Conda environment for subsequent Dockerfile RUN instructions.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy only the files required to build and install phipsurv.
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src

# Dependencies are supplied by ML_env.yml. Disabling build isolation prevents
# pip from downloading a second build environment during the image build.
RUN python -m pip install --no-build-isolation --no-deps .

# Fail the image build early if package, plotting, utility, CLI, or R
# dependencies are missing.
RUN python -c "import phipsurv; import phipsurv.io.data_handler; import phipsurv.plots.helpers; import phipsurv.plots.metrics_helpers; import phipsurv.survival.helpers; import phipsurv.survival.train_test_utils; import phipsurv.utils.peptides_filter; import tabulate; print('phipsurv imports: OK')" \
    && phipsurv -h >/dev/null \
    && Rscript -e "library(maxstat); library(survival); cat('R dependencies: OK\n')"

# Use a writable directory for mounted data, configs, results, and notebooks.
USER root
RUN mkdir -p /workspace /tmp/matplotlib \
    && chown -R $MAMBA_USER:$MAMBA_USER /workspace /tmp/matplotlib
USER $MAMBA_USER
WORKDIR /workspace

EXPOSE 8888

# `docker run phipsurv` shows help. Supplying another command overrides CMD:
#   docker run ... phipsurv -c configs/config.yaml
#   docker run ... jupyter lab --ip=0.0.0.0 --no-browser
CMD ["phipsurv", "-h"]