import numpy as np
from data import bin_array, test_bin
import struct

# a1 = np.array([32767, 16384, 0, -16384, -32767], dtype=np.int16)

# a2 = np.array(a1, dtype=np.float32) / 32767

# print(a1)
# print(a2)

# bin_array = test_bin

print(len(bin_array))

b = bytearray(bin_array)
print(len(b))

f = np.frombuffer(b, dtype=np.float32)

floats =[]
for i in range(0,len(b),4):
    float_value = struct.unpack('<f', b[i:i+4])[0]
    floats.append(float_value)

print(floats)
print(f)

print(f"{len(f)}   {f[0]}")