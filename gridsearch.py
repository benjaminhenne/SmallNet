import subprocess

for i in range(4):
	for j in range(4):
		subprocess.call(['sbatch' , 'gs_smallnet.sh', str(int(800000/(2**(i+1)))), str(2**(i+6)), str(1e-2/(10**j))])		
