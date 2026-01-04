# import pip
# from subprocess import call

# for dist in pip.get_installed_distributions():
#     call("pip install --upgrade " + dist.project_name, shell=True)

from ..src.model.backbone.SwinTransformer import SwinTransformer
import torch

model = SwinTransformer()
x = torch.randn(1, 3, 224, 224)
y = model(x)



