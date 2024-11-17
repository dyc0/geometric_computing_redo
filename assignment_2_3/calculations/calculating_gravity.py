import torch
import tempfile

f_mass = torch.tensor([-1, 2, -3], dtype=torch.float64)

rho = 0.7
W0 = torch.tensor([5, 3], dtype=torch.float64)

f_v = torch.tensor([
    [-0.7, 1.4, -2.1],
    [-0.7, 1.4, -2.1],
], dtype=torch.float64)

f_per_tet_1 = W0[0] * f_v[0]
f_per_tet_2 = W0[1] * f_v[1]

print("=====")
print("Per tet forces")
print(f_per_tet_1)
print(f_per_tet_2)
print("=====")

f_a = f_per_tet_1 + f_per_tet_2
f_b = f_per_tet_1 + f_per_tet_2
f_c = f_per_tet_1 + f_per_tet_2
f_d = f_per_tet_1
f_e = f_per_tet_2

print("=====")
print("Per vertex forces")
print(0.25 * f_a)
print(0.25 * f_b)
print(0.25 * f_c)
print(0.25 * f_d)
print(0.25 * f_e)
print("=====")

f = torch.stack([f_a, f_b, f_c, f_d, f_e], dim=0)

s = ''
    # Convert the tensor to NumPy and write it as text
s+= '[\n'
for row in f.numpy():
    s += '\t[' + ', '.join(map(str, row)) + '],\n'
s += ']\n'

print("=====")
print("Per vertex forces")
print(s)
print("=====")

barycenters = torch.tensor([
    [0.5, 0.75, 1.25],
    [0.25, 0.25, -0.75],
], dtype=torch.float64)

# Makes no sense, we don't care
deformed_barycenters = torch.tensor([
    [1, 2.75, 3.25],
    [0.7, 0.25624, -3.75]
], dtype=torch.float64)

barycenters_diff =  barycenters - deformed_barycenters

print("=====")
print("Barycenters diff")
print(barycenters_diff)
print("=====")


E_1 = torch.sum(f_per_tet_1 * barycenters_diff[0])
E_2 = torch.sum(f_per_tet_2 * barycenters_diff[1])
print("=====")
print("Per tet energy")
print(E_1)
print(E_2)
print(E_1 + E_2)
print("=====")
