set -e
python3.7 setup.py install
mkdir -p brinkman_forchheimer
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/brinkman_forchheimer.py > ./brinkman_forchheimer/dynamic.log
