track:
	sudo nice -n 10 powermetrics --samplers cpu_power,gpu_power,thermal -o powermetrics.txt -f plist -i 1000

cpu:
	papermill try_phi2_torch.ipynb output.ipynb --log-output -p MAX_TOKENS 200 -p USE_MPS False

mps:
	papermill try_phi2_torch.ipynb output.ipynb --log-output -p MAX_TOKENS 200 -p USE_MPS True

mlx:
	papermill try_phi2_mlx.ipynb output.ipynb --log-output -p MAX_TOKENS 200

# jrun:
# 	papermill $(s) output.ipynb --log-output
# jrun: 
# 	jupyter nbconvert --to script "$(s)" --output _notebook_to_run; cat _notebook_to_run.py | grep -v get_ipython > run_this.py; python3 run_this.py