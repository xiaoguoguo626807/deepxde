set -e
python3.7 setup.py install
mkdir -p Poisson_Neumann_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_Neumann_1d.py --static > ./Poisson_Neumann_1d/static.log
