set -e
python3.7 setup.py install
mkdir -p diffusion_reaction_rate
DDE_BACKEND=paddle python3.7 -u examples/pinn_inverse/diffusion_reaction_rate.py --static > ./diffusion_reaction_rate/static_prim.log
