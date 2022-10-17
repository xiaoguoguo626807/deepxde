set -e
python3.7 setup.py install
mkdir -p Helmholtz_Sound_hard_ABC_2d
FLAGS_enable_eager_mode=0 DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Helmholtz_Sound_hard_ABC_2d.py > ./Helmholtz_Sound_hard_ABC_2d/dynamic.log
