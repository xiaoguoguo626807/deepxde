set -e
python3.7 setup.py install
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_Robin_1d.py --static > ./Poisson_Robin_1d/static.log
