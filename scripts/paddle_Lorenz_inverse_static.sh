set -e
python3.7 setup.py install
mkdir -p Lorenz_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/Lorenz_inverse.py --static > ./Lorenz_inverse/static.log
