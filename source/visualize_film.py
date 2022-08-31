import torch
import numpy as np
from source.models.AutoFX import AutoFX
import matplotlib.pyplot as plt
import pylustrator
#pylustrator.start()

CHECKPOINT = "/home/alexandre/logs/dmd_12aoutFeat/lightning_logs/version_1/checkpoints/epoch=19-step=27920.ckpt"
model = AutoFX.load_from_checkpoint(CHECKPOINT)

x = torch.ones((1, 1, 35000))
feat = torch.zeros((1, 48))

cond = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float)
film1_1_modulation_gamma = model.film1_1(cond)[0].detach().numpy().reshape((8, 8))
film1_1_modulation_beta = model.film1_1(cond)[1].detach().numpy().reshape((8, 8))
film1_2_modulation_gamma = model.film1_2(cond)[0].detach().numpy().reshape((8, 8))
film1_2_modulation_beta = model.film1_2(cond)[1].detach().numpy().reshape((8, 8))
film2_1_modulation_gamma = model.film2_1(cond)[0].detach().numpy().reshape((16, 8))
film2_1_modulation_beta = model.film2_1(cond)[1].detach().numpy().reshape((16, 8))
film2_2_modulation_gamma = model.film2_2(cond)[0].detach().numpy().reshape((16, 8))
film2_2_modulation_beta = model.film2_2(cond)[1].detach().numpy().reshape((16, 8))
film3_1_modulation_gamma = model.film3_1(cond)[0].detach().numpy().reshape((16, 16))
film3_1_modulation_beta = model.film3_1(cond)[1].detach().numpy().reshape((16, 16))
film3_2_modulation_gamma = model.film3_2(cond)[0].detach().numpy().reshape((16, 16))
film3_2_modulation_beta = model.film3_2(cond)[1].detach().numpy().reshape((16, 16))

cond = torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.float)
film1_1_delay_gamma = model.film1_1(cond)[0].detach().numpy().reshape((8, 8))
film1_1_delay_beta = model.film1_1(cond)[1].detach().numpy().reshape((8, 8))
film1_2_delay_gamma = model.film1_2(cond)[0].detach().numpy().reshape((8, 8))
film1_2_delay_beta = model.film1_2(cond)[1].detach().numpy().reshape((8, 8))
film2_1_delay_gamma = model.film2_1(cond)[0].detach().numpy().reshape((16, 8))
film2_1_delay_beta = model.film2_1(cond)[1].detach().numpy().reshape((16, 8))
film2_2_delay_gamma = model.film2_2(cond)[0].detach().numpy().reshape((16, 8))
film2_2_delay_beta = model.film2_2(cond)[1].detach().numpy().reshape((16, 8))
film3_1_delay_gamma = model.film3_1(cond)[0].detach().numpy().reshape((16, 16))
film3_1_delay_beta = model.film3_1(cond)[1].detach().numpy().reshape((16, 16))
film3_2_delay_gamma = model.film3_2(cond)[0].detach().numpy().reshape((16, 16))
film3_2_delay_beta = model.film3_2(cond)[1].detach().numpy().reshape((16, 16))

cond = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.float)
film1_1_disto_gamma = model.film1_1(cond)[0].detach().numpy().reshape((8, 8))
film1_1_disto_beta = model.film1_1(cond)[1].detach().numpy().reshape((8, 8))
film1_2_disto_gamma = model.film1_2(cond)[0].detach().numpy().reshape((8, 8))
film1_2_disto_beta = model.film1_2(cond)[1].detach().numpy().reshape((8, 8))
film2_1_disto_gamma = model.film2_1(cond)[0].detach().numpy().reshape((16, 8))
film2_1_disto_beta = model.film2_1(cond)[1].detach().numpy().reshape((16, 8))
film2_2_disto_gamma = model.film2_2(cond)[0].detach().numpy().reshape((16, 8))
film2_2_disto_beta = model.film2_2(cond)[1].detach().numpy().reshape((16, 8))
film3_1_disto_gamma = model.film3_1(cond)[0].detach().numpy().reshape((16, 16))
film3_1_disto_beta = model.film3_1(cond)[1].detach().numpy().reshape((16, 16))
film3_2_disto_gamma = model.film3_2(cond)[0].detach().numpy().reshape((16, 16))
film3_2_disto_beta = model.film3_2(cond)[1].detach().numpy().reshape((16, 16))

film1_1_max = max(np.max(film1_1_delay_beta), np.max(film1_1_delay_gamma),
                  np.max(film1_1_disto_beta), np.max(film1_1_disto_gamma),
                  np.max(film1_1_modulation_beta), np.max(film1_1_modulation_gamma))
film1_1_min = min(np.min(film1_1_delay_beta), np.min(film1_1_delay_gamma),
                  np.min(film1_1_disto_beta), np.min(film1_1_disto_gamma),
                  np.min(film1_1_modulation_beta), np.min(film1_1_modulation_gamma))
film1_2_max = max(np.max(film1_2_delay_beta), np.max(film1_2_delay_gamma),
                  np.max(film1_2_disto_beta), np.max(film1_2_disto_gamma),
                  np.max(film1_2_modulation_beta), np.max(film1_2_modulation_gamma))
film1_2_min = min(np.min(film1_2_delay_beta), np.min(film1_2_delay_gamma),
                  np.min(film1_2_disto_beta), np.min(film1_2_disto_gamma),
                  np.min(film1_2_modulation_beta), np.min(film1_2_modulation_gamma))
film2_1_max = max(np.max(film2_1_delay_beta), np.max(film2_1_delay_gamma),
                  np.max(film2_1_disto_beta), np.max(film2_1_disto_gamma),
                  np.max(film2_1_modulation_beta), np.max(film2_1_modulation_gamma))
film2_1_min = min(np.min(film2_1_delay_beta), np.min(film2_1_delay_gamma),
                  np.min(film2_1_disto_beta), np.min(film2_1_disto_gamma),
                  np.min(film2_1_modulation_beta), np.min(film2_1_modulation_gamma))
film2_2_max = max(np.max(film2_2_delay_beta), np.max(film2_2_delay_gamma),
                  np.max(film2_2_disto_beta), np.max(film2_2_disto_gamma),
                  np.max(film2_2_modulation_beta), np.max(film2_2_modulation_gamma))
film2_2_min = min(np.min(film2_2_delay_beta), np.min(film2_2_delay_gamma),
                  np.min(film2_2_disto_beta), np.min(film2_2_disto_gamma),
                  np.min(film2_2_modulation_beta), np.min(film2_2_modulation_gamma))
film3_1_max = max(np.max(film3_1_delay_beta), np.max(film3_1_delay_gamma),
                  np.max(film3_1_disto_beta), np.max(film3_1_disto_gamma),
                  np.max(film3_1_modulation_beta), np.max(film3_1_modulation_gamma))
film3_1_min = min(np.min(film3_1_delay_beta), np.min(film3_1_delay_gamma),
                  np.min(film3_1_disto_beta), np.min(film3_1_disto_gamma),
                  np.min(film3_1_modulation_beta), np.min(film3_1_modulation_gamma))
film3_2_max = max(np.max(film3_2_delay_beta), np.max(film3_2_delay_gamma),
                  np.max(film3_2_disto_beta), np.max(film3_2_disto_gamma),
                  np.max(film3_2_modulation_beta), np.max(film3_2_modulation_gamma))
film3_2_min = min(np.min(film3_2_delay_beta), np.min(film3_2_delay_gamma),
                  np.min(film3_2_disto_beta), np.min(film3_2_disto_gamma),
                  np.min(film3_2_modulation_beta), np.min(film3_2_modulation_gamma))
fig = plt.figure(11, constrained_layout=True)
axs = fig.subplots(nrows=2, ncols=3)
plt.setp(axs, xticks=[], yticks=[])
axs[0][0].imshow(film1_1_modulation_gamma, vmin=film1_1_min, vmax=film1_1_max)
axs[0][1].imshow(film1_1_modulation_beta, vmin=film1_1_min, vmax=film1_1_max)
axs[0][2].imshow(film1_1_delay_gamma, vmin=film1_1_min, vmax=film1_1_max)
axs[1][0].imshow(film1_1_delay_beta, vmin=film1_1_min, vmax=film1_1_max)
axs[1][1].imshow(film1_1_disto_gamma, vmin=film1_1_min, vmax=film1_1_max)
im = axs[1][2].imshow(film1_1_disto_beta, vmin=film1_1_min, vmax=film1_1_max)
fig.colorbar(im, ax=axs, location='right')


fig = plt.figure(12, constrained_layout=True)
axs = fig.subplots(nrows=2, ncols=3)
plt.setp(axs, xticks=[], yticks=[])
axs[0][0].imshow(film1_2_modulation_gamma, vmin=film1_2_min, vmax=film1_2_max)
axs[0][1].imshow(film1_2_modulation_beta, vmin=film1_2_min, vmax=film1_2_max)
axs[0][2].imshow(film1_2_delay_gamma, vmin=film1_2_min, vmax=film1_2_max)
axs[1][0].imshow(film1_2_delay_beta, vmin=film1_2_min, vmax=film1_2_max)
axs[1][1].imshow(film1_2_disto_gamma, vmin=film1_2_min, vmax=film1_2_max)
im = axs[1][2].imshow(film1_2_disto_beta, vmin=film1_2_min, vmax=film1_2_max)
fig.colorbar(im, ax=axs, location='right')


fig = plt.figure(21, constrained_layout=True)
axs = fig.subplots(nrows=2, ncols=3)
plt.setp(axs, xticks=[], yticks=[])
axs[0][0].imshow(film2_1_modulation_gamma, vmin=film2_1_min, vmax=film2_1_max)
axs[0][1].imshow(film2_1_modulation_beta, vmin=film2_1_min, vmax=film2_1_max)
axs[0][2].imshow(film2_1_delay_gamma, vmin=film2_1_min, vmax=film2_1_max)
axs[1][0].imshow(film2_1_delay_beta, vmin=film2_1_min, vmax=film2_1_max)
axs[1][1].imshow(film2_1_disto_gamma, vmin=film2_1_min, vmax=film2_1_max)
im = axs[1][2].imshow(film2_1_disto_beta, vmin=film2_1_min, vmax=film2_1_max)
fig.colorbar(im, ax=axs, location='right')


fig = plt.figure(22, constrained_layout=True)
axs = fig.subplots(nrows=2, ncols=3)
plt.setp(axs, xticks=[], yticks=[])
axs[0][0].imshow(film2_2_modulation_gamma, vmin=film2_2_min, vmax=film2_2_max)
axs[0][1].imshow(film2_2_modulation_beta, vmin=film2_2_min, vmax=film2_2_max)
axs[0][2].imshow(film2_2_delay_gamma, vmin=film2_2_min, vmax=film2_2_max)
axs[1][0].imshow(film2_2_delay_beta, vmin=film2_2_min, vmax=film2_2_max)
axs[1][1].imshow(film2_2_disto_gamma, vmin=film2_2_min, vmax=film2_2_max)
im = axs[1][2].imshow(film2_2_disto_beta, vmin=film2_2_min, vmax=film2_2_max)
fig.colorbar(im, ax=axs, location='right')


fig = plt.figure(31, constrained_layout=True)
axs = fig.subplots(nrows=2, ncols=3)
plt.setp(axs, xticks=[], yticks=[])
axs[0][0].imshow(film3_1_modulation_gamma, vmin=film3_1_min, vmax=film3_1_max)
axs[0][1].imshow(film3_1_modulation_beta, vmin=film3_1_min, vmax=film3_1_max)
axs[0][2].imshow(film3_1_delay_gamma, vmin=film3_1_min, vmax=film3_1_max)
axs[1][0].imshow(film3_1_delay_beta, vmin=film3_1_min, vmax=film3_1_max)
axs[1][1].imshow(film3_1_disto_gamma, vmin=film3_1_min, vmax=film3_1_max)
im = axs[1][2].imshow(film3_1_disto_beta, vmin=film3_1_min, vmax=film3_1_max)
fig.colorbar(im, ax=axs, location='right')

fig = plt.figure(32, constrained_layout=True)
axs = fig.subplots(nrows=2, ncols=3)
plt.setp(axs, xticks=[], yticks=[])
axs[0][0].imshow(film3_2_modulation_gamma, vmin=film3_2_min, vmax=film3_2_max)
axs[0][1].imshow(film3_2_modulation_beta, vmin=film3_2_min, vmax=film3_2_max)
axs[0][2].imshow(film3_2_delay_gamma, vmin=film3_2_min, vmax=film3_2_max)
axs[1][0].imshow(film3_2_delay_beta, vmin=film3_2_min, vmax=film3_2_max)
axs[1][1].imshow(film3_2_disto_gamma, vmin=film3_2_min, vmax=film3_2_max)
im = axs[1][2].imshow(film3_2_disto_beta, vmin=film3_2_min, vmax=film3_2_max)
fig.colorbar(im, ax=axs, location='right')









plt.show()
