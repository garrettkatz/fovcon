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
        self.weights = tr.nn.Parameter((tr.randn(len(self.idx)) / len(self.idx)))
        # self.biases = (tr.randn(rows*cols*out_channels) / (rows*cols*out_channels)).requires_grad_(True)
        # self.biases = (tr.randn(rows*cols*out_channels) / len(self.idx)).requires_grad_(True)
        self.biases = tr.nn.Parameter(tr.randn(rows*cols*out_channels) / len(self.idx))

    # def parameters(self):
    #     return (self.weights, self.biases)

    @profile
    def forward(self, img):
        # img.shape = (batch size, rows, cols, channels)

        # assign weights to sparse connectivity matrix
        mat = tr.zeros(self.dims_in, self.dims_out)
        mat.view(self.dims_in*self.dims_out)[self.idx] = self.weights
        self.mat = mat

        # matrix-vector multiply
        out = img.reshape(-1, self.dims_in) @ mat
        out = out + self.biases
        # print(img.shape, img.reshape(-1, self.dims_in).shape, mat.shape, self.biases.shape, out.shape)
        out = out.reshape(-1, self.rows, self.cols, self.out_channels)
        # print(out.shape)
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
    # pt.show()

    rows, cols, in_channels = 4, 4, 1
    out_channels, kernel_size = 1, 1
    mod = ConvMat(rows, cols, in_channels, out_channels, kernel_size)

    mod.weights.data[:] = 1
    mod.biases.data[:] = 0

    out = mod(tr.tensor(img).to(tr.float32)[None,:,:,None])
    out.sum().backward()

    print(list(mod.parameters()))
    print("weight, bias grads:")
    print(mod.weights.grad)
    print(mod.biases.grad)

    pt.imshow(out.detach().numpy()[:,:,0])
    # pt.show()

    ## confluence pattern
    
    dim = 7
    kws = [1, 2, 3, 4]
    cmats = [conv_mask(dim, dim, 1, 1, kw).reshape(dim**2, dim**2) for kw in kws]
    
    idx = np.array([0,1,2,3,2,1,0])
    midx = np.minimum(idx[:, np.newaxis], idx)

    pixdrop = np.ones((dim, dim), dtype=bool)
    pixdrop[0, ::2] = 0
    pixdrop[-1, ::2] = 0
    pixdrop[::2, 0] = 0
    pixdrop[::2, -1] = 0

    pixdrop[1, ::3] = 0
    pixdrop[-2, ::3] = 0
    pixdrop[::3, 1] = 0
    pixdrop[::3, -2] = 0

    pt.imshow(pixdrop, vmin=0, vmax=1)
    # pt.show()
    
    confluence = np.zeros(cmats[0].shape)
    for i, (r,c) in enumerate(it.product(range(dim), repeat=2)):
        confluence[i] = cmats[midx[r,c]][i]
    
    # pt.figure(figsize=(26,4.5))
    pt.figure(figsize=(20,3.5))
    
    for n, kw in enumerate(kws):
        pt.subplot(1, len(kws)+3, n+1)
        pt.imshow(cmats[n])
        pt.title(f"Kernel radius {kw}")
    
    pt.subplot(1, len(kws)+3, len(kws)+1)
    pt.title("Variable")
    pt.imshow(confluence)

    pt.subplot(1, len(kws)+3, len(kws)+2)
    pt.title("Pixel drop-out")
    # pt.imshow(cmats[-2][:,pixdrop.flatten()])
    pt.imshow(cmats[-2] * pixdrop.reshape(1, -1))
    
    pt.subplot(1, len(kws)+3, len(kws)+3)
    pt.imshow(midx.reshape(-1, 1)*np.ones((1,5)))
    
    pt.tight_layout()
    pt.savefig('confluence_pattern.png')
    # pt.show()
    
    
