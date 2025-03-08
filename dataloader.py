import os
import numpy as np
import matplotlib.pyplot as pt

class DataLoader:
    def __init__(self, path, trial_bases):

        # load and parse the set of available "actions" (buttons)
        with open(os.path.join(path, "action_enums.txt"), "r") as f: lines = f.readlines()
        
        self.action_enum = {-1: None} # action_enum[n] will store the nth available action
        for line in lines:
            if "=" not in line: continue
            action, num = line.split("=")
            action = action.strip()
            num = int(num.strip())
            self.action_enum[num] = action

        self.path = path
        self.trial_bases = trial_bases

    def examples(self):

        for trial_base in self.trial_bases:

            with open(os.path.join(self.path, f"{trial_base}.txt"), "r") as f:
                label_info = f.readlines()

            for t in range(1, len(label_info)):

                # extract action and other label information
                fields = label_info[t].strip().split(",")
                frame_id, episode_id, score, duration, reward, action = fields[:6]

                # actions: integer key in action_enum
                if action == "null": action = -1
                else: action = int(action)

                # raw image
                img = pt.imread(os.path.join(self.path, trial_base, f"{frame_id}.png")).astype(float)

                # gaze
                # according to https://zenodo.org/records/3451402
                # gaze = x0,y0,...,xn,yn, x is horizontal, y is vertical, (0,0) is top left
                # reformat to [..., [row, col], ...] array
                if fields[6] == "null":
                    gaze = ()
                else:
                    xy = np.array(tuple(map(float, fields[6:]))).reshape(-1, 2)
                    gaze = xy[:,[1,0]] # x is column, y is row

                yield action, gaze, img

# helper to filter out some frames
def filter_frames(examples):
    for (action, gaze, img) in examples:
    
        # skip frames when no action was taken
        if action == -1: continue
    
        # skip frames with only one gaze point
        # if action == 2 or len(gaze) <= 1: continue
        if len(gaze) <= 1: continue
    
        # skip frames where gaze points barely changed
        if np.fabs(gaze[0] - gaze[-1]).max() < 5: continue
    
        # # skip frames where gaze coordinates are out of bounds
        if (gaze < 0).any() or (gaze >= img.shape[:2]).any(): continue

        # keep the remainder
        yield (action, gaze, img)

if __name__ == "__main__":

    # path to atarihead data
    data_path = os.path.join(os.environ["HOME"], "atarihead")

    # the leading portion of filenames for one trial
    trial_base = "100_RZ_3592991_Aug-24-11-44-38" # venture
    # trial_base = "52_RZ_2394668_Aug-10-14-52-42" # ms pacman

    # init data loader
    dl = DataLoader(data_path, [trial_base])

    # init fcg
    from fcg import FovealCartesianGeometry
    rho_0 = 20
    rho_max = 60 # cols are 160
    numrings = 40
    fcg = FovealCartesianGeometry(rho_0, rho_max, numrings)
    
    # extract and plot a few arbitrary frames from the trial
    pt.figure(figsize=(25,10))
    sp_col = 0
    last_t = 0
    for t, (action, gaze, img) in enumerate(filter_frames(dl.examples())):
    
        # skip at least 50 frames each time to get more variety
        if t - last_t <= 50: continue
    
        # save the time-step of most recently displayed frame so you can skip the next 50
        print(t)
        print(gaze)
        last_t = t
    
        # plot the gaze coordinates on the frame
        pt.subplot(2,5,sp_col+1)
        pt.imshow(img)
        for g in range(len(gaze)-1):
            pt.plot(gaze[g:g+1,0], gaze[g:g+1,1], 'w.-')
        pt.plot(gaze[-1,0], gaze[-1,1], 'go')
        pt.title(f"Frame {t}")
    
        # extract and show the FCG subsample of the current frame
        gc, gr = gaze[-1].round().astype(int)
        foveal = fcg.sample(img, gr, gc)
    
        pt.subplot(2,5,5+sp_col+1)
        pt.imshow(foveal)
        pt.title(f"{dl.action_enum[action]}")
    
        sp_col += 1
        if sp_col == 5: break

    print("foveal shape", foveal.shape)

    pt.show()
