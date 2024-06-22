import numpy as np
from time import perf_counter
from convmat import ConvMat
from scipy.special import lambertw
import matplotlib.pyplot as pt
import torch as tr

# whether kernel weights are shared
# when false, hidden features for parity is roughly linear from 0 to 50
shared = True

img_dim = 79 # fcg size after preproc
R = img_dim // 2 # max radius

# max kernel size
K = 20 # rho_0 from preproc

# area under exponential drop-off curve
# areas should equal integral of K*np.exp(-decay_rate*r) from r = 0 to R
areas = np.linspace(0, R*K, 11)[1:-1]

# solve for decay rates that realize given areas, based on examples in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lambertw.html
a = K / areas
b = -a
c = -R
decay_rates = np.real(a - lambertw(-b*c*np.exp(a*c))/c)

# decay_rate = -np.log((min_ks+1) / (max_ks+1)) / rad
# powers = np.array([.01, .05, .1, .5, 1, 5, 10, 100])

r = np.arange(0,R+1,0.1)
for decay_rate in decay_rates:
    fr = K * np.exp(-decay_rate * r)
    portion = fr.mean() / K
    pt.plot(r, fr, '-', label=f"$\\lambda = {np.real(decay_rate):.3f}$")
    # pt.plot(r, fr, '-', label=f"{portion:.3f}")
# pt.plot(np.arange(rad+1), max_ks * np.exp(-decay_rate * np.arange(rad+1)), 'r-')
# for power in powers:
#     pt.plot(np.arange(rad+1), max_ks * np.exp(-power * decay_rate * np.arange(rad+1)), '-', label=str(power))
pt.plot(r, [K]*len(r), 'k:', label='full con')

pt.xlabel("Radius from fovea")
pt.ylabel("Kernel size")
pt.legend()
pt.savefig('kernel_configs.png')
pt.show()

# number of parameters

class ConvModel(tr.nn.Module):
    def __init__(self, hid_channels, decay_rate, sparse=True, shared=True):
        super().__init__()

        # based on FCG and atari data
        rows = cols = 79
        in_channels = 3
        out_dim = 18
        K = 20

        i = np.arange(rows) - (rows//2)
        r = (i[:,None]**2 + i**2)**0.5
        kernel_sizes = np.maximum(1, (K * np.exp(-decay_rate * r)).round().astype(int))
        # print(kernel_sizes)
        # pt.subplot(1,2,1)
        # pt.imshow(kernel_sizes)
        # pt.colorbar()
        # pt.subplot(1,2,2)
        # pt.imshow(r)
        # pt.colorbar()
        # pt.show()

        self.conv = ConvMat(rows, cols, in_channels, hid_channels, kernel_sizes, sparse, shared)
        self.relu = tr.nn.LeakyReLU()
        self.flat = tr.nn.Flatten()
        self.lin = tr.nn.Linear(rows * cols * hid_channels, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flat(x)
        x = self.lin(x)
        return x

# run configs
if True:

    configs = []
    mats = []
    for hid_channels in [1]:
        for dr, decay_rate in enumerate(decay_rates):
    
            for sparse in [False, True]:
    
                model = ConvModel(hid_channels, decay_rate, sparse, shared)
                num_params = len(tr.nn.utils.parameters_to_vector(model.parameters()))
    
                num_reps = 3
                rep_times = []
                for reps in range(num_reps):
                    start = perf_counter()
                    inp = tr.randn(4, 79, 79, 3)
                    out = model(inp)
                    rep_times.append(perf_counter() - start)
                mintime = min(rep_times)
    
                print(f"{dr}: decay={decay_rate:.3f}, hid={hid_channels}, sparse={sparse}: {num_params} params, {mintime}s forward")
                configs.append((decay_rate, hid_channels, sparse, num_params, mintime))

                if not sparse:
                    mat = model.conv.mat.detach().numpy()
                    print(mat[:len(mat)//3].shape)
                    mats.append(mat[:len(mat)//3] != 0)

    results = np.array(configs).T

    np.save(f"kernel_configs_shared_{shared}.npy", results)
    np.save(f"kernel_configs_mats_shared_{shared}.npy", np.stack(mats))

results = np.load(f"kernel_configs_shared_{shared}.npy")
mats = np.load(f"kernel_configs_mats_shared_{shared}.npy")

fig = pt.figure(figsize=(6, 6))
for d in range(len(mats)):
    pt.subplot(3, 3, d+1)
    pt.imshow(mats[d].T)
    pt.axis('off')
    pt.title(f"$\\lambda = {decay_rates[d]:.3f}$")
fig.supxlabel("input neuron")
fig.supylabel("hidden neuron")
pt.tight_layout()
pt.savefig("kernel_configs_mats.png")
pt.show()

pt.figure(figsize=(10,4))

pt.subplot(1,3,1)
pt.plot(results[0, results[2]==False], results[3, results[2]==False], 'ko-')
pt.xlabel("Decay rate")
pt.ylabel("Trainable Parameters")

pt.subplot(1,3,2)
pt.plot(results[0, results[2]==False], (results[3, -1] / results[3, results[2]==False]).round(), 'ko-')
pt.xlabel("Decay rate")
pt.ylabel("Hidden features for parity")

pt.subplot(1,3,3)
pt.plot(results[0, results[2]==False], results[4, results[2]==False], 'bo-', label="Full matrix")
pt.plot(results[0, results[2]==True], results[4, results[2]==True], 'ro-', label="Sparse matrix")
pt.xlabel("Decay rate")
pt.ylabel("Forward time (3-pass min)")
pt.legend()

pt.tight_layout()
pt.savefig('kernel_configs_stats.png')
pt.show()
            
