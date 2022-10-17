set -e
python3.7 setup.py install
mkdir -p Helmholtz_Sound_hard_ABC_2d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Helmholtz_Sound_hard_ABC_2d.py --static > ./Helmholtz_Sound_hard_ABC_2d/static.log
