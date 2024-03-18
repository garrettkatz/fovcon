import os
import torch as tr
import matplotlib.pyplot as pt
from dataloader import DataLoader, filter_frames
from fcg import FovealCartesianGeometry

if __name__ == "__main__":

    do_preproc = True

    # path to atarihead data
    data_path = os.path.join(os.environ["HOME"], "atarihead")

    # the leading portion of filenames for one trial
    trial_base = "100_RZ_3592991_Aug-24-11-44-38" # venture
    # trial_base = "52_RZ_2394668_Aug-10-14-52-42" # ms pacman

    # save name
    save_name = os.path.join(data_path, trial_base) + ".pt"

    # init fcg
    rho_0 = 20
    rho_max = 60 # cols are 160
    numrings = 40
    fcg = FovealCartesianGeometry(rho_0, rho_max, numrings)

    # init data loader
    dl = DataLoader(data_path, [trial_base])

    # extract samples
    if do_preproc:

        inputs = []
        targets = []
        for ex, (action, gaze, img) in enumerate(filter_frames(dl.examples())):
    
            # preprocess example
            gc, gr = gaze[-1].round().astype(int)
            inp = fcg.sample(img, gr, gc)
            inp = tr.tensor(inp).to(tr.float32)
    
            # add to batch
            inputs.append(inp)
            targets.append(action)
    
            print(f"{ex} examples")
            # if ex == 40: break
    
        inputs = tr.stack(inputs)
        targets = tr.tensor(targets)

        tr.save((inputs, targets), save_name)

    inputs, targets = tr.load(save_name)

    print('inputs, targets shape')
    print(inputs.shape, targets.shape)

    idx = tr.randperm(len(inputs))[:20]    
    for sp, i in enumerate(idx):
        pt.subplot(4,5, sp+1)
        pt.imshow(inputs[i].numpy())
    pt.show()
