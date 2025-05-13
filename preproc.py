import os
import torch as tr
import matplotlib.pyplot as pt
from dataloader import DataLoader, filter_frames
from fcg import FovealCartesianGeometry

if __name__ == "__main__":

    do_preproc = False

    # path to atarihead data
    # data_path = os.path.join(os.environ["HOME"], "atarihead")
    data_path = os.path.join(os.environ["HOME"], "atarihead/venture")

    # the leading portion of filenames for one trial
    # venture
    trial_base = "100_RZ_3592991_Aug-24-11-44-38"
    # trial_base = "101_RZ_3603032_Aug-24-14-31-37"
    # trial_base = "107_RZ_3682314_Aug-25-12-32-42"
    # trial_base = "111_RZ_3865406_Aug-27-15-24-49"
    # trial_base = "114_RZ_3870288_Aug-27-16-47-08"
    # trial_base = "127_JAW_2764190_Dec-08-14-18-45"
    # trial_base = "130_JAW_3029229_Dec-11-15-54-52"
    # trial_base = "133_JAW_3106946_Dec-12-13-30-02"
    # trial_base = "154_KM_5790027_Jan-12-14-48-42"
    # trial_base = "155_KM_5791179_Jan-12-15-08-15"
    # trial_base = "391_RZ_2354543_Jul-05-14-23-36"
    # trial_base = "400_RZ_2533918_Jul-07-16-12-38"
    # trial_base = "437_RZ_3132888_Jul-14-14-35-28"
    # # ms pacman
    # trial_base = "52_RZ_2394668_Aug-10-14-52-42"

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
            gr, gc = gaze[-1].astype(int)
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

    pt.figure(figsize=(4,8))
    idx = tr.randperm(len(inputs))[:8]
    for sp, i in enumerate(idx):
        pt.subplot(4, 2, sp+1)
        pt.imshow(inputs[i].numpy())
        pt.title(f"Action {targets[i].item()}")
        pt.axis("off")
    pt.savefig(os.environ["HOME"] + "/nexus/grants/nsf_braid_qinru/venture_fcg.png")
    pt.show()

