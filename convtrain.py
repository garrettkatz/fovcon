import pickle as pk
import matplotlib.pyplot as pt
import torch as tr
from fcg import FovealCartesianGeometry
from dataloader import DataLoader, filter_frames
from convmat import ConvMat

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

if __name__ == "__main__":

    do_train = True

    # the leading portion of filenames for one trial
    trial_base = "100_RZ_3592991_Aug-24-11-44-38"

    # init data loader
    dl = DataLoader(".", [trial_base])

    # init fcg
    print("init fcg...")
    rho_0 = 6
    rho_max = 60 # raw cols are 160
    numrings = 20
    fcg = FovealCartesianGeometry(rho_0, rho_max, numrings)

    # init model
    print("init model...")
    dim = fcg.out_dim
    model = tr.nn.Sequential(
        ConvMat(dim, dim, in_channels=3, out_channels=1, kernel_size=3),
        tr.nn.LeakyReLU(),
        tr.nn.Flatten(),
        tr.nn.Linear(dim**2, max(dl.action_enum.keys())+1),
    )

    # init optimizer and loss
    opt = tr.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = tr.nn.CrossEntropyLoss()

    # training loop
    if do_train:
        loss_curve = []
        for epoch in range(30):
            examples = filter_frames(dl.examples())
            for b, (inputs, targets) in enumerate(batched(16, fcg, examples)):
    
                # forward pass
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss_curve.append(loss.item())
    
                # gradient update
                loss.backward()
                opt.step()
                opt.zero_grad()
    
                print(f"epoch {epoch}, update {b}: loss = {loss_curve[-1]}")
    
            #     if len(loss_curve) == 10: break
            # if len(loss_curve) == 10: break

        with open("convtrain.pkl", "wb") as f:
            pk.dump(loss_curve, f)

    with open("convtrain.pkl", "rb") as f:
        loss_curve = pk.load(f)

    pt.plot(loss_curve)
    pt.show()

