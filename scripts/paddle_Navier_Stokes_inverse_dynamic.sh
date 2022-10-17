set -e
python3.7 setup.py install
mkdir -p Navier_Stokes_inverse
FLAGS_enable_eager_mode=0 DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/Navier_Stokes_inverse.py > ./Navier_Stokes_inverse/dynamic.log
