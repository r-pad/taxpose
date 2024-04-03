#!/bin/bash

# Set the display env variable
export DISPLAY=:99
export VGL_DISPLAY="egl"

# Start Xvfb
Xvfb $DISPLAY -screen 0 1024x768x24 -nolisten tcp -nolisten unix +extension GLX  > /dev/null 2>&1 &

# Wait a little for Xvfb to start
sleep 2

echo "Xvfb started on display $DISPLAY"

# Run your OpenGL application with VirtualGL (add +v command if needed.)
vglrun \
    -d egl0 \
    -ld /opt/baeisner/.pyenv/versions/3.9.12/lib/python3.9/site-packages/torch/lib \
    "$@"
