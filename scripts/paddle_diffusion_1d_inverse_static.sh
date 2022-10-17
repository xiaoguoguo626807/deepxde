set -e
python3.7 setup.py install
mkdir -p diffusion_1d_inverse
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/diffusion_1d_inverse.py --static > ./diffusion_1d_inverse/static.log
