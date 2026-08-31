# set DAMASK_ROOT to your DAMASK installation's parent directory before running
DAMASK_ROOT="${DAMASK_ROOT}"

/bin/cp -rf ./mod_src/* ${DAMASK_ROOT}/damask-3.0.0-alpha4/src/

cmake -S ${DAMASK_ROOT}/damask-3.0.0-alpha4 -B ${DAMASK_ROOT}/build-grid -DDAMASK_SOLVER=grid
sudo cmake --build ${DAMASK_ROOT}/build-grid --target install
