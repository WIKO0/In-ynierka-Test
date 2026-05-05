
import torch
import pytorch3d

print("Torch:", torch.__version__)
print("PyTorch3D:", pytorch3d.__version__)

print("CUDA dostępne:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "brak")


from pytorch3d.structures import Meshes

# prosty trójkąt
verts = torch.tensor([[[0,0,0],[1,0,0],[0,1,0]]], dtype=torch.float32)
faces = torch.tensor([[[0,1,2]]])

mesh = Meshes(verts=verts, faces=faces)
print("Mesh OK:", mesh)