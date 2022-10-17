set -e
python3.7 setup.py install
mkdir -p ode_2nd
FLAGS_enable_eager_mode=0 DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/ode_2nd.py > ./ode_2nd/dynamic.log
