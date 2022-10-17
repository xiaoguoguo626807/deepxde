set -e
python3.7 setup.py install
mkdir -p ode_system
FLAGS_enable_eager_mode=0 DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/ode_system.py > ./ode_system/dynamic.log
