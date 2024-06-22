try:
    profile
except:
    profile = lambda x: x
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
import torch as tr

def conv_mask(rows, cols, in_channels, out_channels, kernel_sizes):
    """
    kernel_sizes should be a (rows, cols) int array of kernel sizes at each pixel
    """

    dims = (rows, cols, out_channels, rows, cols, in_channels)
    mask = np.zeros(dims, dtype=int) # nonzero connections
    kszs = np.empty(dims, dtype=int) # unraveled kernel sizes
    kidx = np.empty(dims, dtype=int) # unraveled kernel indices

    # number of weights per kernel position
    chan = in_channels * out_channels

    # populate sparsity structure
    for (i_out, j_out) in it.product(range(rows), range(cols)):

        kernel_size = kernel_sizes[i_out, j_out]
        for ki, (di, dj) in enumerate(it.product(range(-kernel_size, kernel_size+1), repeat=2)):

            # current out coords
            i_in, j_in = i_out + di, j_out + dj

            # skip out-of-bounds
            if not (0 <= i_in < rows): continue
            if not (0 <= j_in < cols): continue

            # mask connections within kernel
            mask[i_out, j_out, :, i_in, j_in, :] = 1

            # save kernel size and unraveled weight index within kernel
            kszs[i_out, j_out, :, i_in, j_in, :] = kernel_size
            kidx[i_out, j_out, :, i_in, j_in, :].flat = ki*chan + np.arange(chan)

    return mask, kszs, kidx

class ConvMat(tr.nn.Module):
    def __init__(self, rows, cols, in_channels, out_channels, kernel_sizes, sparse=True, shared=False):
        super().__init__()

        # save dimensions
        self.dims = rows, cols, in_channels, out_channels, kernel_sizes
        self.dims_in = rows*cols*in_channels
        self.dims_out = rows*cols*out_channels
        self.rows = rows
        self.cols = cols
        self.out_channels = out_channels

        # save flag for sparse vs dense matrix multiply
        # and for shared kernel weights
        self.sparse = sparse
        self.shared = shared

        # set up conv mask
        mask, kszs, kidx = conv_mask(*self.dims)

        # set up flat index of non-zero connections, transposed to work well with batch input
        self.idx = np.flatnonzero(mask.T)

        # set up i,j index of trainable weights (also transposed)
        self.ij = np.stack(np.unravel_index(self.idx, (self.dims_in, self.dims_out)))

        if shared:

            # build the offsets in weight vector for each kernel size
            unique_sizes = np.unique(kernel_sizes)
            offset = np.cumsum(in_channels*out_channels * (2*unique_sizes+1)**2)

            # convert to lookup dictionaries
            offset = {sz: offset[i] for (i,sz) in enumerate(unique_sizes)}
            lookup = {sz: i for (i,sz) in enumerate(unique_sizes)}

            # kernel entry index of shared weights
            # self.widx[i] is the index of the kernel entry shared by the ith connection
            self.widx = tr.tensor([kidx.T.flat[i] + offset[kszs.T.flat[i]] for i in self.idx])

            # index for shared biases based on kernel size at (r,c) and output channel d
            self.bidx = tr.tensor([
                lookup[kernel_sizes[r,c]]*out_channels + d
                for (r,c,d) in it.product(range(rows),range(cols),range(out_channels))])

            # one weight per distinct kernel entry
            num_w = max(self.widx)+1

            # one bias per kernel size per output channel
            assert max(self.bidx)+1 == len(unique_sizes) * out_channels
            num_b = max(self.bidx)+1

        else:

            # one weight per connection
            num_w = len(self.idx)

            # one bias per output
            num_b = rows*cols*out_channels
    
        # init trainable parameter tensors
        self.weights = tr.nn.Parameter((tr.randn(num_w) / num_w))
        self.biases = tr.nn.Parameter(tr.randn(num_b) / num_w) # bias on same order of magnitude as weights

    @profile
    def forward(self, img):
        # img.shape = (batch size, rows, cols, channels)

        # expand parameters if shared at multiple indices
        weights, biases = self.weights, self.biases
        if self.shared:
            weights = weights[self.widx]
            biases = biases[self.bidx]

        ## assign weights to connectivity matrix
        # sparse case
        if self.sparse:
            mat = tr.sparse_coo_tensor(self.ij, weights, size=(self.dims_in, self.dims_out))

        # dense case
        else:
            mat = tr.zeros(self.dims_in, self.dims_out)
            mat.view(self.dims_in*self.dims_out)[self.idx] = weights

        # save connectivity matrix for future reference if needed
        self.mat = mat

        # reshaped matrix-vector multiply
        out = img.reshape(-1, self.dims_in) @ mat
        out = out + biases
        out = out.reshape(-1, self.rows, self.cols, self.out_channels)
        return out


if __name__ == "__main__":

    shared = False

    img = np.arange(4).reshape(4, 1) * np.ones(4)
    img[1:3, 1:3] += 4
    mask, kszs, kidx = conv_mask(4, 4, 3, 3, np.full((4,4), 1))
    cmat = mask.reshape(16*3, 16*3)
    kmat = (mask * (kidx + 1)).reshape(16*3, 16*3)
    smat = (mask * kszs).reshape(16*3, 16*3)
    
    pt.figure(figsize=(14,4))
    pt.subplot(1,5,1)
    pt.imshow(cmat)
    pt.subplot(1,5,2)
    pt.imshow(kmat)
    pt.subplot(1,5,3)
    pt.imshow(smat)
    pt.subplot(1,5,4)
    pt.imshow(img.reshape(-1, 1), vmax=2*4)
    pt.subplot(1,5,5)
    pt.imshow(img, vmax=2*4)
    pt.savefig("cmat_ravel.png")
    pt.show()

    rows, cols, in_channels = 4, 4, 1
    out_channels, kernel_size = 1, 1
    mod = ConvMat(rows, cols, in_channels, out_channels, np.full((rows, cols), kernel_size), sparse=True, shared=shared)

    mod.weights.data[:] = 1
    mod.biases.data[:] = 0

    out = mod(tr.tensor(img).to(tr.float32)[None,:,:,None])
    out.sum().backward()

    print(list(mod.parameters()))
    print("weight, bias grads:")
    print(mod.weights.grad)
    print(mod.biases.grad)

    print(f"\n{mod.weights.numel()+mod.biases.numel()} parameters total\n")

    # make sure result doesn't change for non-sparse version
    mod_dense = ConvMat(rows, cols, in_channels, out_channels, np.full((rows, cols), kernel_size), sparse=False, shared=shared)

    mod_dense.weights.data[:] = 1
    mod_dense.biases.data[:] = 0

    out_dense = mod_dense(tr.tensor(img).to(tr.float32)[None,:,:,None])
    out_dense.sum().backward()

    assert tr.allclose(out, out_dense)
    assert tr.allclose(mod.weights.grad, mod_dense.weights.grad)
    assert tr.allclose(mod.biases.grad, mod_dense.biases.grad)

    pt.imshow(out.detach().numpy()[:,:,0])
    # pt.show()
    pt.close()

    ## confluence pattern
    
    dim = 7
    kws = [1, 2, 3, 4]
    cmats = [
        conv_mask(dim, dim, 1, 1, np.full((dim,dim), kw))[0].reshape(dim**2, dim**2)
        for kw in kws]
    
    idx = np.array([0,1,2,3,2,1,0])
    midx = np.minimum(idx[:, np.newaxis], idx)
    kernel_sizes = np.array(kws)[midx]
    print('kernel sizes:')
    print(kernel_sizes)

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
    pt.close()
    
    confluence = np.zeros(cmats[0].shape)
    for i, (r,c) in enumerate(it.product(range(dim), repeat=2)):
        confluence[i] = cmats[midx[r,c]][i]

    mask, kszs, kidx = conv_mask(dim, dim, 1, 1, kernel_sizes)
    confluence_mat = mask.reshape(dim**2, dim**2)
    assert (confluence == confluence_mat).all()
    
    # pt.figure(figsize=(26,4.5))
    pt.figure(figsize=(20,3.5))
    
    for n, kw in enumerate(kws):
        pt.subplot(1, len(kws)+4, n+1)
        pt.imshow(cmats[n])
        pt.title(f"Kernel radius {kw}")
    
    pt.subplot(1, len(kws)+4, len(kws)+1)
    pt.title("Variable")
    pt.imshow(confluence)

    pt.subplot(1, len(kws)+4, len(kws)+2)
    pt.title("Variable")
    pt.imshow(confluence_mat)

    pt.subplot(1, len(kws)+4, len(kws)+3)
    pt.title("Pixel drop-out")
    # pt.imshow(cmats[-2][:,pixdrop.flatten()])
    pt.imshow(cmats[-2] * pixdrop.reshape(1, -1))
    
    pt.subplot(1, len(kws)+4, len(kws)+4)
    pt.imshow(midx.reshape(-1, 1)*np.ones((1,5)))
    
    pt.tight_layout()
    pt.savefig('confluence_pattern.png')
    pt.show()

    kmat = (mask * (kidx + 1)).reshape(dim**2, dim**2)
    smat = (mask * kszs).reshape(dim**2, dim**2)

    pt.subplot(1,3,1)
    pt.title("Connectivity")
    pt.imshow(confluence_mat)
    pt.subplot(1,3,2)
    pt.title("kernel indices")
    pt.imshow(kmat)
    pt.subplot(1,3,3)
    pt.title("kernel sizes")
    pt.imshow(smat)
    pt.show()
    
