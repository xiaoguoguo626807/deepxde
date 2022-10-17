set -e
python3.7 setup.py install
mkdir -p ode_system
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/ode_system.py --static > ./ode_system/static.log
