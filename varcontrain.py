try:
    profile
except:
    profile = lambda x: x
import os
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
import torch as tr
from fcg import FovealCartesianGeometry
from dataloader import DataLoader, filter_frames
from convmat import ConvMat

@profile
def batched(batch_size, inputs, targets):
    split_inputs = tr.split(inputs, batch_size)
    split_targets = tr.split(targets, batch_size)
    yield from zip(split_inputs, split_targets)

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


@profile
def main():

    do_train = True
    # num_epochs = 20 # not shared
    num_epochs = 60 # shared
    decay_rate = .256
    hid_channels = 1
    batch_size = 32
    learning_rate = .01#0.0003
    hid_dim = 128 #3*out_dim

    # path to atarihead data
    data_path = os.path.join(os.environ["HOME"], "atarihead")

    # the leading portion of filenames for one trial
    trial_base = "100_RZ_3592991_Aug-24-11-44-38"

    # load preprocessed data
    inputs, targets = tr.load(os.path.join(data_path, trial_base) + ".pt")

    # init model
    print("init model...")
    model = ConvModel(hid_channels, decay_rate, sparse=True, shared=True)

    print("Total parameter count:")
    print(tr.nn.utils.parameters_to_vector(model.parameters()).numel())

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
                # logits = model(inp, show=(b==0))
                logits = model(inp)
                # if b == 0: print(logits)
                loss = loss_fn(logits, targ)
                loss_curve.append(loss.item())
                correct.append((logits.argmax(dim=-1) == targ).to(float).mean())
    
                # gradient update
                loss.backward()
                opt.step()
                opt.zero_grad()
    
                print(f"epoch {epoch}, update {b}: loss = {loss_curve[-1]}")
    
            #     if len(loss_curve) == 5: break
            # if len(loss_curve) == 5: break

            accu_curve.append(np.mean(correct))
            print(f"epoch {epoch}: accu = {accu_curve[-1]}")

        with open("varcontrain.pkl", "wb") as f:
            pk.dump((loss_curve, accu_curve), f)

    with open("varcontrain.pkl", "rb") as f:
        loss_curve, accu_curve = pk.load(f)

    nparams = len(tr.nn.utils.parameters_to_vector(model.parameters()))
    updates_per_epoch = len(loss_curve) / len(accu_curve)
    print(f"{nparams} parameters")

    fig = pt.figure(figsize=(6,3))
    pt.subplot(1,2,1)
    pt.plot(2*np.arange(len(loss_curve))*nparams, loss_curve)
    pt.ylabel("Loss")
    # pt.xlabel("Total mult-adds")
    # pt.yscale('log')
    pt.subplot(1,2,2)
    pt.plot(2*np.arange(len(accu_curve))*nparams*updates_per_epoch, accu_curve)
    pt.ylabel("Accuracy")
    # pt.xlabel("Total mult-adds")
    # pt.xlabel("Epoch (pass over data)")
    fig.supxlabel("Total multiply-adds (updates x params)")
    pt.tight_layout()
    pt.savefig("varcontrain.png")
    pt.show()

if __name__ == "__main__": main()

