import torch
import tempfile

PK1 = torch.tensor([
    [
        [  781.2,   0.8,    0.0],
        [ -390.4, 196.8,    0.0],
        [ 0.0,      0.0,  100.8]
    ],
    [
        [ 1600.8,   1.6,    0.0],
        [ -800.0, 403.2,    0.0],
        [ 0.0,      0.0,  801.6]
    ]
], dtype=torch.float64)

W0 = torch.tensor([5.0, 3.0], dtype=torch.float64)

D = torch.tensor([
    [
        [  0.0,  2.0,  0.0],
        [  0.0,  0.0,  3.0],
        [ -5.0, -5.0, -5.0]
    ],
    [
        [ -1.0, 2.0, 0.0],
        [ -2.0, 0.0, 3.0],
        [ -3.0, 0.0, 0.0]
    ]
], dtype=torch.float64)

J1 = torch.inverse(D[0])
print(J1)
print(J1.T)
J2 = torch.inverse(D[1])
print(J2)
assert torch.allclose(torch.matmul(D[0], J1), torch.eye(3, dtype=torch.float64))
assert torch.allclose(torch.matmul(D[1], J2), torch.eye(3, dtype=torch.float64))
print("=====")


H1 = -W0[0] * torch.matmul(PK1[0], J1.T)
H2 = -W0[1] * torch.matmul(PK1[1], J2.T)

print("=====")
print(H1)
print("=====")
print(H2)
print("=====")
print("=====")

f_a = H1[:, 0] - torch.sum(H2, dim=1)
f_b = H1[:, 1] + H2[:, 1]
f_c = H1[:, 2] + H2[:, 2]
f_d = -torch.sum(H1, dim=1)
f_e = H2[:, 0]

F = torch.stack([f_a, f_b, f_c, f_d, f_e], dim=1).T
assert F.dtype == torch.float64
print(F)
with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w') as tmp_file:
    # Convert the tensor to NumPy and write it as text
    for row in F.numpy():
        tmp_file.write(' '.join(map(str, row)) + '\n')
    print(f"F saved to temporary text file: {tmp_file.name}")
