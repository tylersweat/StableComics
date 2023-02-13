## Environment setup for sending batch Attend and Excite jobs

If the `miniconda3` module is not yet loaded, run:
`module load miniconda3`

Create the latent diffusion model environment, `ldm`, by running:
`conda env create -f environment.yaml`

Add the following to `~/.bashrc`

```bash
# Activate ldm conda environment.
conda activate ldm

# Add LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/tysweat0/.conda/envs/ldm/lib
```

Run `StableComics/AttendExcite/save_model.py` to download the model locally. The compute nodes do not allow access to the Internet.