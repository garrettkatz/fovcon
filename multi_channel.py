try:
    profile
except:
    profile = lambda x: x
import os
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
    def __init__(self, k_p, c_p, k_f, c_f):
        # kernel size and channels for periph and fovea
        super().__init__()
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
        p = self.periph(x)
        f = self.foveal(x[:,20:60,20:60,:])
        return self.lin(tr.cat((p,f), dim=-1))
        

if __name__ == "__main__":

    do_train = True
    data_path = os.path.join(os.environ["HOME"], "atarihead")
    trial_base = "100_RZ_3592991_Aug-24-11-44-38"
    num_epochs = 100
    batch_size = 16
    learning_rate = 0.001

    # load preprocessed data
    inputs, targets = tr.load(os.path.join(data_path, trial_base) + ".pt")

    # init model
    # in: (batch_size, 79, 79, 3)

    # model = tr.nn.Sequential(
    #     Permute((0,3,1,2)), # (batch, 3, 79, 79)
    #     tr.nn.Conv2d(3, 1, 6),
    #     tr.nn.Flatten(),
    #     tr.nn.LeakyReLU(),
    #     tr.nn.Linear(5476, 18),
    # )

    # model = MultiChannel(3, 1, 3, 1)
    model = constant_channel(3, 2)

    nparams = len(tr.nn.utils.parameters_to_vector(model.parameters()))
    print(f"{nparams} parameters")

    # init optimizer and loss
    opt = tr.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = tr.nn.CrossEntropyLoss()

    # training loop
    if do_train:
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
    
                if b % 40 == 0: print(f"epoch {epoch}, update {b}: loss = {loss_curve[-1]}")
    
            #     if len(loss_curve) == 5: break
            # if len(loss_curve) == 5: break

            accu_curve.append(np.mean(correct))
            print(f"epoch {epoch}: accu = {accu_curve[-1]}")

        with open("multi_channel.pkl", "wb") as f:
            pk.dump((loss_curve, accu_curve), f)

    with open("multi_channel.pkl", "rb") as f:
        loss_curve, accu_curve = pk.load(f)

    print(f"{nparams} parameters")
    updates_per_epoch = len(loss_curve) / len(accu_curve)

    fig = pt.figure(figsize=(6,3))
    pt.subplot(1,2,1)
    pt.plot(2*np.arange(len(loss_curve)), loss_curve)
    pt.ylabel("Loss")
    pt.subplot(1,2,2)
    pt.plot(2*np.arange(len(accu_curve))*updates_per_epoch, accu_curve)
    pt.ylabel("Accuracy")
    fig.supxlabel("Parameter updates")
    pt.tight_layout()
    pt.savefig("multi_channel.png")
    pt.show()


