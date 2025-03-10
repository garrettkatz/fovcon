try:
    profile
except:
    profile = lambda x: x
import os
import itertools as it
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr

@profile
def batched(batch_size, inputs, targets):
    split_inputs = tr.split(inputs, batch_size)
    split_targets = tr.split(targets, batch_size)
    yield from zip(split_inputs, split_targets)

class Permute(tr.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return tr.permute(x, self.dims)

def constant_channel(k, c):
    hid = c * (79 - k + 1)**2
    return tr.nn.Sequential(
        Permute((0,3,1,2)),
        tr.nn.Conv2d(3, c, k),
        tr.nn.Flatten(),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(hid, 18)
    )

class MultiChannel(tr.nn.Module):
    def __init__(self, *hparams):
        super().__init__()

        # kernel size and channels for periph and fovea
        (k_p, c_p), (k_f, c_f) = hparams

        if min(k_p, c_p) > 0:
            self.periph = tr.nn.Sequential(
                Permute((0,3,1,2)),
                tr.nn.Conv2d(3, c_p, k_p),
                tr.nn.Flatten(),
                tr.nn.LeakyReLU(),
            )

        if min(k_f, c_f) > 0:
            self.foveal = tr.nn.Sequential(
                Permute((0,3,1,2)),
                tr.nn.Conv2d(3, c_f, k_f),
                tr.nn.Flatten(),
                tr.nn.LeakyReLU(),
            )

        hid_p = c_p * (79 - k_p + 1)**2
        hid_f = c_f * (40 - k_f + 1)**2

        self.lin = tr.nn.Linear(hid_p + hid_f, 18)

    def forward(self, x):
        pf = []
        if hasattr(self, "periph"): pf.append( self.periph(x) )
        if hasattr(self, "foveal"): pf.append( self.foveal(x[:,20:60,20:60,:]) )
        return self.lin(tr.cat(pf, dim=-1))
        

if __name__ == "__main__":

    do_train = False
    data_path = os.path.join(os.environ["HOME"], "atarihead")
    trial_base = "100_RZ_3592991_Aug-24-11-44-38"
    num_reps = 1
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.01
    hparam_max = 7

    # setup hyperparameters
    kc = [(0,0)] + list(it.product(range(1, hparam_max), repeat=2))
    kc_both = list(it.product(kc, repeat=2))[1:] # omit both zerod
    # print("hparam settings:")
    # for hparams in kc_both: print(hparams)

    # load preprocessed data
    inputs, targets = tr.load(os.path.join(data_path, trial_base) + ".pt")

    # experimental runs
    results = []
    if do_train:

        for (rep, hparams) in it.product(range(num_reps), kc_both):

            # init model
            model = MultiChannel(*hparams)
            nparams = len(tr.nn.utils.parameters_to_vector(model.parameters()))
            print(f"rep {rep} hparams {hparams}: {nparams} parameters")
        
            # init optimizer and loss
            opt = tr.optim.Adam(model.parameters(), lr=learning_rate)
            loss_fn = tr.nn.CrossEntropyLoss()
        
            # training loop
            loss_curve = []
            accu_curve = []
            for epoch in range(num_epochs):
                correct = []
                for b, (inp, targ) in enumerate(batched(batch_size, inputs, targets)):
    
                    # forward pass
                    logits = model(inp)
                    loss = loss_fn(logits, targ)
                    loss_curve.append(loss.item())
                    correct.append((logits.argmax(dim=-1) == targ).to(float).mean())
        
                    # gradient update
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
        
                    # if b % 40 == 0: print(f"epoch {epoch}, update {b}: loss = {loss_curve[-1]}")
    
                accu_curve.append(np.mean(correct))
                print(f" epoch {epoch}: accu = {accu_curve[-1]}")

            results.append((hparams, nparams, loss_curve, accu_curve))

            with open("multi_channel.pkl", "wb") as f: pk.dump(results, f)

    with open("multi_channel.pkl", "rb") as f: results = pk.load(f)

    # (hparams, nparams, loss_curve, accu_curve) = results[0]
    # updates_per_epoch = len(loss_curve) / len(accu_curve)
    # regret = 1.0 - np.mean(accu_curve)
    # print(f"{nparams} parameters, regret={regret}")

    # fig = pt.figure(figsize=(6,3))
    # pt.subplot(1,2,1)
    # pt.plot(loss_curve)
    # pt.ylabel("Loss")
    # pt.xlabel("Parameter update")
    # pt.subplot(1,2,2)
    # pt.plot(accu_curve)
    # pt.ylabel("Accuracy")
    # pt.xlabel("Epoch")
    # # fig.supxlabel("Parameter updates")
    # pt.tight_layout()
    # pt.savefig("multi_channel.png")
    # pt.show()

    # single pathways

    regrets = {"foveal": {}, "peripheral": {}}
    for (hparams, nparams, loss_curve, accu_curve) in results:
        (k_p, c_p), (k_f, c_f) = hparams

        if k_p == c_p == 0:
            if k_f not in regrets["foveal"]: regrets["foveal"][k_f] = {}
            regrets["foveal"][k_f][c_f] = 1.0 - np.mean(accu_curve)
        elif k_f == c_f == 0:
            if k_p not in regrets["peripheral"]: regrets["peripheral"][k_p] = {}
            regrets["peripheral"][k_p][c_p] = 1.0 - np.mean(accu_curve)
        else: continue

    pt.figure(figsize=(8,4))
    for p, pathway in enumerate(("foveal", "peripheral")):
        pt.subplot(1,2,p+1)
        pt.title(pathway)
        for k in regrets[pathway]:
            c, reg = zip(*regrets[pathway][k].items())
            pt.plot(c, reg, label=f"k={k}")
        if p == 0: pt.ylabel("Regret")
        pt.ylim([-.1, 1.1])
        pt.legend()
    pt.gcf().supxlabel("Feature Channels")
    pt.tight_layout()
    pt.savefig(os.environ["HOME"] + "/nexus/grants/nsf_braid_qinru/single_channel.png")
    pt.show()

    pt.figure(figsize=(12,4))
    npar, regret, ratio = [], [], []
    for (hparams, nparams, loss_curve, accu_curve) in results:
        
        # subsample points
        if np.random.rand() < .65: continue
        
        (k_p, c_p), (k_f, c_f) = hparams

        npar.append(nparams)
        regret.append(1.0 - np.mean(accu_curve))
        ratio.append(c_f*k_f**2 / (c_f*k_f**2 + c_p*k_p**2))

    pt.scatter(npar, regret, marker='o', c=ratio, edgecolors="k")
    pt.xlabel("Model Parameters")
    pt.ylabel("Training Regret")
    pt.colorbar(label="Foveal Parameter Ratio")
    pt.tight_layout()
    pt.savefig(os.environ["HOME"] + "/nexus/grants/nsf_braid_qinru/multi_channel.png")    
    pt.show()

