import numpy as np

def f(x) -> float:   
    return float(x)/32767.0

a = [-32767,-16384,0,16384,32767]

an = np.array(a, dtype=np.int16)

f = np.array( [float(ani)/32767.0 for ani in an], dtype=np.float32)

print(a)
print(f)