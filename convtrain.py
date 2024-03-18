try:
    profile
except:
    profile = lambda x: x
import os
import pickle as pk
import matplotlib.pyplot as pt
import torch as tr
from fcg import FovealCartesianGeometry
from dataloader import DataLoader, filter_frames
from convmat import ConvMat

@profile
def batched(batch_size, fcg, examples):
    inputs, targets = [], []
    for (action, gaze, img) in examples:

        # preprocess example
        gc, gr = gaze[-1].round().astype(int)
        inp = fcg.sample(img, gr, gc)
        inp = tr.tensor(inp).to(tr.float32)

        # add to batch
        inputs.append(inp)
        targets.append(action)

        # continue until batch size reached
        if len(inputs) < batch_size: continue

        # yield current batch and start new one
        yield tr.stack(inputs), tr.tensor(targets)
        inputs, targets = [], []

@profile
def main():

    do_train = False
    num_epochs = 20
    kernel_size = 3
    batch_size = 32
    learning_rate = 0.0005
    hid_dim = 128 #3*out_dim

    # path to atarihead data
    data_path = os.path.join(os.environ["HOME"], "atarihead")

    # the leading portion of filenames for one trial
    trial_base = "100_RZ_3592991_Aug-24-11-44-38"

    # init data loader
    dl = DataLoader(data_path, [trial_base])

    # init fcg
    print("init fcg...")
    rho_0 = 6
    rho_max = 60 # raw cols are 160
    numrings = 20
    fcg = FovealCartesianGeometry(rho_0, rho_max, numrings)

    # init model
    print("init model...")
    dim = fcg.out_dim
    out_dim = max(dl.action_enum.keys())+1
    model = tr.nn.Sequential(
        ConvMat(dim, dim, in_channels=3, out_channels=1, kernel_size=kernel_size),
        tr.nn.LeakyReLU(),
        tr.nn.Flatten(),
        tr.nn.Linear(dim**2, hid_dim),
        tr.nn.LeakyReLU(),
        tr.nn.Linear(hid_dim, out_dim),
    )

    # init optimizer and loss
    opt = tr.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = tr.nn.CrossEntropyLoss()

    # training loop
    if do_train:
        loss_curve = []
        for epoch in range(num_epochs):
            examples = filter_frames(dl.examples())
            for b, (inputs, targets) in enumerate(batched(batch_size, fcg, examples)):
    
                # forward pass
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss_curve.append(loss.item())
    
                # gradient update
                loss.backward()
                opt.step()
                opt.zero_grad()
    
                print(f"epoch {epoch}, update {b}: loss = {loss_curve[-1]}")
    
            #     if len(loss_curve) == 5: break
            # if len(loss_curve) == 5: break

        with open("convtrain.pkl", "wb") as f:
            pk.dump(loss_curve, f)

    with open("convtrain.pkl", "rb") as f:
        loss_curve = pk.load(f)

    pt.plot(loss_curve)
    pt.show()

if __name__ == "__main__": main()

