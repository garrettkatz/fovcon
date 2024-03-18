try:
    profile
except:
    profile = lambda x: x
import numpy as np
import matplotlib.pyplot as pt

####
#### build the foveal cartesian geometry
#### https://www.researchgate.net/publication/221492348_A_New_Foveal_Cartesian_Geometry_Approach_used_for_Object_Tracking
####

class FovealCartesianGeometry:
    def __init__(self, rho_0, rho_max, numrings):

        # set up the rings
        periph = numrings - rho_0
        a = np.exp(np.log(rho_max / rho_0) / periph)
        rhos = (rho_0 * a**np.arange(1, periph+1)).round().astype(int)
        rhos = np.concatenate((np.arange(rho_0), rhos))

        # set up the indices
        self.dst_idx = {} # per-ring destination flat indices
        self.src_coords = {} # per-ring row,col coordinates
        
        # superimposed masks for all rings (sanity check)
        self.src_masks = np.zeros((2*rho_max+1, 2*rho_max+1))

        # compute indices for each ring
        for r, rho in enumerate(rhos):
        
            # row/col start/stop ordinates in FCG result (inclusive)
            start, stop = numrings-1-r, numrings-1+r
        
            # sampled row/col ordinates in raw image
            samples = np.linspace(-rho, rho, 2*r + 1).round().astype(int)
        
            # mask out source and destination pixels
            dst_mask = np.zeros((2*numrings-1, 2*numrings-1))
            dst_mask[[start, stop], start:stop+1] = 1
            dst_mask[start:stop+1, [start, stop]] = 1

            src_mask = np.zeros((2*rho+1, 2*rho+1))
            src_mask[[[0], [-1]], rho+samples] = 1
            src_mask[rho+samples,  [[0], [-1]]] = 1
        
            self.src_masks[[[rho_max - rho], [rho_max + rho]], rho_max+samples] = 1
            self.src_masks[rho_max+samples, [[rho_max - rho], [rho_max + rho]]] = 1

            # extract indices from masks
            self.dst_idx[r] = np.flatnonzero(dst_mask) # flat index
            self.src_coords[r] = np.argwhere(src_mask) # row/col index

        # save parameters for sampling
        self.numrings = numrings
        self.rhos = rhos

        # output dimension
        self.out_dim = 2*self.numrings-1

    @profile
    def sample(self, img, fr, fc):

        # rgb channels of destination image
        numrings = self.numrings
        result = np.zeros((2*numrings-1, 2*numrings-1, 3))

        # central pixel
        idx_0 = (3*self.dst_idx[0][:,None] + np.array([[0,1,2]])).flatten()
        result.flat[idx_0] = img[fr,fc,:]

        # concentric rings
        for r in range(1, len(self.rhos)):
            rho = self.rhos[r]

            # foveal-offset source coordinates
            f_coords = (self.src_coords[r] - rho) + (fr, fc)

            # pooling width (radial distance to previous ring)
            width = rho - self.rhos[r-1]

            # allocate average-pooled pixel values
            pools = np.zeros((len(self.dst_idx[r]), 3))
            idx_r = (3*self.dst_idx[r][:,None] + np.array([[0,1,2]])).flatten()

            # pool around each foveal-offset pixel
            for p, (row, col) in enumerate(f_coords):

                # truncate pooling boundaries to image dimensions
                row_lo = max(0, row - width + 1)
                col_lo = max(0, col - width + 1)
                if row_lo >= img.shape[0] or col_lo >= img.shape[1]: continue
    
                row_hi = min(row + width, img.shape[0])
                col_hi = min(col + width, img.shape[1])
                if row_hi < 0 or col_hi < 0: continue

                # skip empty slices
                if row_lo == row_hi or col_lo == col_hi: continue

                # calculate pooled average
                pools[p] = img[row_lo:row_hi, col_lo:col_hi, :].mean(axis=(0,1))

            # # build sparse version of pooling
            # offsets = np.array(it.product(range(-width+1, width), repeat=2)) # (offsets, 2)
            # pool_coords = f_coords + offsets # (centers, 1, 2) + (1, offsets, 2) = (centers, offsets, 2)
            # np.ravel_multi_index((pool_coords[:,:,0], pool_coords[:,:,1]), img.shape[:2] 
            # indices = np.arange(len(f_coords))[:,
            # smat = sparse_coo_tensor(indices, values=1, size=(len(idx_r), img.numel()))

            # assign pooled pixel values into destination indices
            result.flat[idx_r] = pools

        # return stacked rgb channels
        return result



if __name__ == "__main__":

    rho_0 = 6
    rho_max = 60 # cols are 160
    numrings = 20

    fcg = FovealCartesianGeometry(rho_0, rho_max, numrings)

    print(fcg.dst_idx[0])

    # visualize the FCG sample coordinates in source image
    pt.figure(figsize =(10,10))
    pt.imshow(fcg.src_masks)
    pt.title('FCG sampling')
    # pt.savefig('fcg_src_mask.png')
    pt.show()

