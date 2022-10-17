set -e
python3.7 setup.py install
mkdir -p Navier_Stokes_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/Navier_Stokes_inverse.py --static > ./Navier_Stokes_inverse/static.log
