set -e
python3.7 setup.py install
mkdir -p Lorenz_inverse
FLAGS_enable_eager_mode=0 DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/Lorenz_inverse.py > ./Lorenz_inverse/dynamic.log
