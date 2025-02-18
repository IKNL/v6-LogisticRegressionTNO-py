# basic python3 image as base
FROM harbor2.vantage6.ai/algorithms/algorithm-base

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="v6-logistic-regression-py"

RUN pip install mpyc==0.7
RUN python -m pip install 'tno.mpc.mpyc.secure_learning[gmpy]'
COPY tno.mpc.protocols.secure_inner_join-dev-py3-none-any.whl .
RUN pip install tno.mpc.protocols.secure_inner_join-dev-py3-none-any.whl


RUN apt update
RUN apt install iperf3 -y
RUN apt install iputils-ping -y

# install federated algorithm
COPY . /app
RUN pip install /app

# In Vantage6 versions 3.1+, you can use VPN communication between algorithms
# over multiple ports. You can specify the ports that are allowed for
# communication here, along with a label that helps you identify them.
# If no ports are specified (and VPN is available), only port 8888 is available
# by default
EXPOSE 8888
LABEL p8888 'mpc_runtime'

# EXPOSE 8080
# LABEL p8080 'tno_com'

ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `docker_wrapper()` when the image is run.
CMD python -c "from vantage6.tools.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"
