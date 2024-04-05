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
    def __init__(self, in_dim, hid_dim, out_dim, kernel_size):
        super().__init__()
        # self.conv = ConvMat(in_dim, in_dim, in_channels=3, out_channels=1, kernel_sizes=np.full((in_dim, in_dim), kernel_size))
        self.conv = ConvMat(
            in_dim, in_dim, in_channels=3, out_channels=1,
            kernel_sizes=np.random.randint(1, kernel_size+1, (in_dim, in_dim)))
        self.relu = tr.nn.LeakyReLU()
        self.flat = tr.nn.Flatten()
        # self.lin1 = tr.nn.Linear(in_dim**2, hid_dim)
        # self.lin2 = tr.nn.Linear(hid_dim, out_dim)
        self.lin = tr.nn.Linear(in_dim**2, out_dim)
    def forward(self, x, show=False):
        if show:
            pt.subplot(1,3,1)
            pt.imshow(x[0].detach())
        x = self.conv(x)
        # print('conv', x.shape, x.reshape(x.shape[0], -1)[:10,:10])
        if show:
            pt.subplot(1,3,2)
            pt.imshow(x[0].detach())
            pt.subplot(1,3,3)
            pt.imshow(self.conv.mat.detach())
            pt.show()
        x = self.relu(x)
        x = self.flat(x)
        # # print('flat', x.shape, x.reshape(x.shape[0], -1)[:10,:10])
        # x = self.lin1(x)
        # x = self.relu(x)
        # # print('lin1', x.shape, x.reshape(x.shape[0], -1)[:10,:10])
        # x = self.lin2(x)
        # # print('lin2', x.shape, x.reshape(x.shape[0], -1)[:10,:10])
        # # input('.')

        x = self.lin(x)

        return x

@profile
def main():

    do_train = True
    num_epochs = 10
    kernel_size = 3
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
    dim = inputs.shape[1]
    out_dim = 18 # number of actions
    # model = tr.nn.Sequential(
    #     ConvMat(dim, dim, in_channels=3, out_channels=8, kernel_size=kernel_size),
    #     tr.nn.LeakyReLU(),
    #     ConvMat(dim, dim, in_channels=8, out_channels=1, kernel_size=kernel_size),
    #     # ConvMat(dim, dim, in_channels=3, out_channels=1, kernel_size=kernel_size),
    #     tr.nn.LeakyReLU(),
    #     tr.nn.Flatten(),
    #     tr.nn.Linear(dim**2, hid_dim),
    #     tr.nn.LeakyReLU(),
    #     tr.nn.Linear(hid_dim, out_dim),
    # )
    model = ConvModel(dim, hid_dim, out_dim, kernel_size)

    # # fully-connected
    # model = tr.nn.Sequential(
    #     tr.nn.Flatten(),
    #     tr.nn.Linear(3*dim**2, hid_dim),
    #     tr.nn.LeakyReLU(),
    #     tr.nn.Linear(hid_dim, out_dim),
    # )

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

        with open("convtrain.pkl", "wb") as f:
            pk.dump((loss_curve, accu_curve), f)

    with open("convtrain.pkl", "rb") as f:
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
    pt.savefig("train.png")
    pt.show()

if __name__ == "__main__": main()

