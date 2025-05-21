import sys
from . import cli
import os
import pynvml


# Initialiser pynvml
pynvml.nvmlInit()

# Récupérer le nombre de GPU
device_count = pynvml.nvmlDeviceGetCount()

# Récupérer la mémoire utilisée pour chaque GPU
gpu_mem_used = []
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_mem_used.append((i, meminfo.used))

# Trouver le GPU avec le moins de mémoire utilisée
gpu_least_used = min(gpu_mem_used, key=lambda x: x[1])[0]

# Fixer CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_least_used)

# Affichage optionnel
print(f"Utilisation du GPU {gpu_least_used} (le moins chargé)")

cim10_cli = cli.Cim10Cli()

if __name__ == "__main__":
    cim10_cli.run(argv=sys.argv[1:])
