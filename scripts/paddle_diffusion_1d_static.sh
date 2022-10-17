set -e
python3.7 setup.py install
mkdir -p diffusion_1d
DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/diffusion_1d.py --static > ./diffusion_1d/static.log
