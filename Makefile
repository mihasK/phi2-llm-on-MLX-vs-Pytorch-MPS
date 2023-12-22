track:
	sudo nice -n 10 powermetrics --samplers cpu_power,gpu_power,thermal -o powermetrics.txt -f plist -i 1000
