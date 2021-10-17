FROM nvcr.io/nvidia/pytorch:21.08-py3
# Create a working directory
RUN mkdir /app
WORKDIR /app

RUN conda install -c conda-forge poetry

RUN pip uninstall torch torchvision torchtext -y

COPY poetry.lock .
COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
     && poetry install --no-interaction --no-ansi

USER developer

EXPOSE 8888

CMD ["bash", "-c", "jupyter lab"]
