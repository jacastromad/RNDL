xhost +
docker run -e DISPLAY=:0 -v ./:/opt/project -v /tmp/.X11-unix:/tmp/.X11-unix --network host --rm --gpus all --workdir /opt/project -i -t rndl_pytorch python $1
xhost -

