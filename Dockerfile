FROM ubuntu:latest

# get system dependencies
RUN apt-get update && apt-get install -y git python3 python3-pip

# mount workspace
COPY . /workspace
WORKDIR /workspace
VOLUME [ "/workspace" ]

# install python dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir \
    black pipreqs \
    numpy pandas \
    torch torchvision torchaudio \
    jupyter jupyterlab jupyter_contrib_nbextensions \
    --break-system-packages

# dev: find out requirements
RUN rm -rf requirements.txt
RUN pipreqs .

# deploy: install requirements
# RUN pip install -r requirements.txt

# run jupyter notebook server
RUN pip install --no-cache-dir jupyter jupyterlab jupyter_contrib_nbextensions --break-system-packages
ENV JUPYTER_ENABLE_LAB=yes
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''", "--ServerApp.allow_origin='*'", "--ServerApp.disable_check_xsrf=True"]
EXPOSE 8888
