import os
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from dataloader import DataLoader, filter_frames
from fcg import FovealCartesianGeometry

if __name__ == "__main__":

    data_path = os.path.join(os.environ["HOME"], "atarihead/venture")
    # trial_base = "100_RZ_3592991_Aug-24-11-44-38"
    trial_base = "101_RZ_3603032_Aug-24-14-31-37"

    # init data loader
    dl = DataLoader(data_path, [trial_base])

    idx = np.random.choice(range(100), 8).tolist()
    all_dists = []
    for ex, (action, gaze, img) in enumerate(filter_frames(dl.examples())):
        print(ex)
        # print(ex, gaze)

        dists = np.sum((gaze[1:] - gaze[:-1])**2, axis=1)**0.5
        all_dists.append(dists)

        if ex in idx:
            pt.subplot(2,len(idx)//2,1+idx.index(ex))
            pt.imshow(img)
            pt.plot(gaze[:,1], gaze[:,0], 'bo-')
            # pt.plot([10], [20], 'bo')
            pt.axis("off")
            print(gaze)
            # pt.show()
            # input('.')

        if ex == 100: break

    pt.show()

    stdevs = [np.std(dists) for dists in all_dists]
    pt.hist(stdevs)
    pt.show()

    pt.hist([len(dists) for dists in all_dists])
    pt.xlabel("Number of timesteps per frame")
    pt.ylabel("Count in dataset")
    pt.show()

    idx = np.argmax(stdevs)
    for ex, (action, gaze, img) in enumerate(filter_frames(dl.examples())):
        if ex == idx: break

    dists = np.sum((gaze[1:] - gaze[:-1])**2, axis=1)**0.5
    print(np.std(dists))

    pt.subplot(1,2,1)
    pt.plot(dists)
    pt.xlabel("t")
    pt.ylabel("||g(t+1) - g(t)||")

    pt.subplot(1,2,2)
    pt.imshow(img)
    pt.plot(gaze[:,1], gaze[:,0], 'bo-')
    pt.axis("off")
    pt.show()

    # all_dists = np.concatenate(all_dists)
    # pt.hist(all_dists)
    # pt.show()


