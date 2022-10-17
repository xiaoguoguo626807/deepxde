set -e
python3.7 setup.py install
mkdir -p ode_2nd
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/ode_2nd.py --static > ./ode_2nd/static.log
