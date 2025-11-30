#!/bin/bash

CONTAINER_NAME="cic-live"

# Check if container exists (running or stopped) and remove if so
if [ "$(sudo docker ps -a -q -f name=$CONTAINER_NAME)" ]; then
  echo "Removing existing container $CONTAINER_NAME..."
  sudo docker rm -f $CONTAINER_NAME
fi

# Run the new container
sudo docker run --rm --net=host --cap-add=NET_ADMIN --cap-add=NET_RAW -v /tmp/captures:/tmp/captures --name $CONTAINER_NAME cicflowmeter-live
