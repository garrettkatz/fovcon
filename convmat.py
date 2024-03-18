try:
    profile
except:
    profile = lambda x: x
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import torch as tr

def conv_mask(rows, cols, in_channels, out_channels, kernel_size):
    dims = (rows, cols, out_channels, rows, cols, in_channels)

    # populate mask for sparsity structure
    mask = np.zeros(dims, dtype=int)
    for (i_in, j_in) in it.product(range(rows), range(cols)):
        for (di, dj) in it.product(range(-kernel_size, kernel_size+1), repeat=2):

            # current out coords
            i_out, j_out = i_in + di, j_in + dj

            # skip out-of-bounds
            if not (0 <= i_out < rows): continue
            if not (0 <= j_out < cols): continue

            # mask connections within kernel
            mask[i_out, j_out, :, i_in, j_in, :] = 1

    return mask

class ConvMat(tr.nn.Module):
    def __init__(self, rows, cols, in_channels, out_channels, kernel_size):
        super().__init__()

        # save dimensions
        self.dims = (rows, cols, in_channels, out_channels, kernel_size)
        self.dims_in = rows*cols*in_channels
        self.dims_out = rows*cols*out_channels
        self.rows = rows
        self.cols = cols
        self.out_channels = out_channels

        # set up flat index of trainable weights
        mask = conv_mask(*self.dims)
        self.idx = np.flatnonzero(mask.T)

        # init trainable parameter tensors
        self.weights = (tr.randn(len(self.idx)) / len(self.idx)).requires_grad_(True)
        self.biases = (tr.randn(rows*cols*out_channels) / (rows*cols*out_channels)).requires_grad_(True)

    def parameters(self):
        return (self.weights, self.biases)

    @profile
    def forward(self, img):
        # img.shape = (batch size, rows, cols, channels)

        # assign weights to sparse connectivity matrix
        mat = tr.zeros(self.dims_in, self.dims_out)
        mat.view(self.dims_in*self.dims_out)[self.idx] = self.weights

        # matrix-vector multiply
        out = img.reshape(-1, self.dims_in) @ mat + self.biases
        out = out.reshape(-1, self.rows, self.cols, self.out_channels)
        return out


if __name__ == "__main__":

    img = np.arange(4).reshape(4, 1) * np.ones(4)
    img[1:3, 1:3] += 4
    mask = conv_mask(4, 4, 3, 3, 1)
    cmat = mask.reshape(16*3, 16*3)
    
    pt.figure(figsize=(14,4))
    pt.subplot(1,3,1)
    pt.imshow(cmat)
    pt.subplot(1,3,2)
    pt.imshow(img.reshape(-1, 1), vmax=2*4)
    pt.subplot(1,3,3)
    pt.imshow(img, vmax=2*4)
    pt.savefig("cmat_ravel.png")
    pt.show()

    rows, cols, in_channels = 4, 4, 1
    out_channels, kernel_size = 1, 1
    mod = ConvMat(rows, cols, in_channels, out_channels, kernel_size)

    mod.weights.data[:] = 1
    mod.biases.data[:] = 0

    out = mod(tr.tensor(img).to(tr.float32)[None,:,:,None])
    out.sum().backward()

    print(mod.biases.grad)

    pt.imshow(out.detach().numpy()[:,:,0])
    pt.show()

    ## confluence pattern
    
    dim = 7
    kws = [1, 2, 3, 4]
    cmats = [conv_mask(dim, dim, 1, 1, kw).reshape(dim**2, dim**2) for kw in kws]
    
    idx = np.array([0,1,2,3,2,1,0])
    midx = np.minimum(idx[:, np.newaxis], idx)
    print(midx)
    
    confluence = np.zeros(cmats[0].shape)
    for i, (r,c) in enumerate(it.product(range(dim), repeat=2)):
        confluence[i] = cmats[midx[r,c]][i]
    
    pt.figure(figsize=(26,4.5))
    
    for n, kw in enumerate(kws):
        pt.subplot(1, len(kws)+2, n+1)
        pt.imshow(cmats[n])
        pt.title(f"Kernel radius {kw}")
    
    pt.subplot(1, len(kws)+2, len(kws)+1)
    pt.title("Confluent")
    pt.imshow(confluence)
    
    pt.subplot(1, len(kws)+2, len(kws)+2)
    pt.imshow(midx.reshape(-1, 1)*np.ones((1,5)))
    
    pt.tight_layout()
    pt.savefig('confluence_pattern.png')
    pt.show()
    
    
