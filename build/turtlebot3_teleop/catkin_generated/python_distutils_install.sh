#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/drcl_yang/ros/tennis_trajectroy/src/turtlebot3_teleop"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/drcl_yang/ros/tennis_trajectroy/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/drcl_yang/ros/tennis_trajectroy/install/lib/python3/dist-packages:/home/drcl_yang/ros/tennis_trajectroy/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/drcl_yang/ros/tennis_trajectroy/build" \
    "/usr/bin/python3" \
    "/home/drcl_yang/ros/tennis_trajectroy/src/turtlebot3_teleop/setup.py" \
     \
    build --build-base "/home/drcl_yang/ros/tennis_trajectroy/build/turtlebot3_teleop" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/drcl_yang/ros/tennis_trajectroy/install" --install-scripts="/home/drcl_yang/ros/tennis_trajectroy/install/bin"