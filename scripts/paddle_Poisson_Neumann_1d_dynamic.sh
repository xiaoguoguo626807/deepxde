set -e
python3.7 setup.py install
CUDA_VISIBLE_DEVICES=1 DDE_BACKEND=paddle python3.7 -u examples/pinn_forward/Poisson_Neumann_1d.py > ./Poisson_Neumann_1d/dynamic.log
