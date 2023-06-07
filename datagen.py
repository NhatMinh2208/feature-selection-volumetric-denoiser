import os
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import random
import uuid
import vfo
from scipy import ndimage
from random import randint
mi.set_variant("cuda_ad_rgb")
patch_size = 64 # patches are 64x64
n_patches = 400

def read_and_render(src) -> mi.Bitmap:
    scene = mi.load_file(src)
    img = mi.render(scene)
    bmp =  mi.Bitmap(img, channel_names=['GT.R', 'GT.G', 'GT.B', 'color.R', 'color.G', 'color.B',
                                          'Lss.R','Lss.G', 'Lss.B', "L_ms.R", "L_ms.G", "L_ms.B",
                     "albedo_ss.R", "albedo_ss.G", "albedo_ss.B", "albedo_ms.R", "albedo_ms.G", "albedo_ms.B",
                    "sigma_s_ss.R", "sigma_s_ss.G", "sigma_s_ss.B", "sigma_s_ms.R", "sigma_s_ms.G", "sigma_s_ms.B",
                     "sigma_t_ss.R", "sigma_t_ss.G", "sigma_t_ss.B", "sigma_t_ms.R", "sigma_t_ms.G", "sigma_t_ms.B",  
                     "position_ss.X", "position_ss.Y", "position_ss.Z",  "position_ms.X", "position_ms.Y", "position_ms.Z",
                     "T_ss.R","T_ss.G", "T_ss.B", "T_ms.R","T_ms.G", "T_ms.B",
                     "tau_ss.R", "tau_ss.G", "tau_ss.B", "tau_ms.R", "tau_ms.G", "tau_ms.B",
                     "z_cam.T",  "z_v_ss.T", "z_v_ms.T"])
    return bmp

def read_exr(src) -> mi.Bitmap: 
    bmp_exr = mi.Bitmap(src)
    return bmp_exr

def write_exr(data : dict, target):
    bmp = mi.Bitmap(np.concatenate((data['GT'],
                           data['color'],
                           data['Lss'],
                           data['L_ms'],
                           data['albedo_ss'],
                           data['albedo_ms'],
                           data['sigma_s_ss'],
                           data['sigma_s_ms'],
                           data['sigma_t_ss'],
                           data['sigma_t_ms'],
                           data['position_ss'],
                           data['position_ms'],
                           data['T_ss'],
                           data['T_ms'],
                           data['tau_ss'],
                           data['tau_ms'],
                           data['z_cam'],
                           data['z_v_ss'],
                           data['z_v_ms']
                          ), axis=2)  , channel_names=['GT.R', 'GT.G', 'GT.B', 'color.R', 'color.G', 'color.B',
                                                        'Lss.R','Lss.G', 'Lss.B', "L_ms.R", "L_ms.G", "L_ms.B",
                     "albedo_ss.R", "albedo_ss.G", "albedo_ss.B", "albedo_ms.R", "albedo_ms.G", "albedo_ms.B",
                    "sigma_s_ss.R", "sigma_s_ss.G", "sigma_s_ss.B", "sigma_s_ms.R", "sigma_s_ms.G", "sigma_s_ms.B",
                     "sigma_t_ss.R", "sigma_t_ss.G", "sigma_t_ss.B", "sigma_t_ms.R", "sigma_t_ms.G", "sigma_t_ms.B",  
                     "position_ss.X", "position_ss.Y", "position_ss.Z",  "position_ms.X", "position_ms.Y", "position_ms.Z",
                     "T_ss.R","T_ss.G", "T_ss.B", "T_ms.R","T_ms.G", "T_ms.B",
                     "tau_ss.R", "tau_ss.G", "tau_ss.B", "tau_ms.R", "tau_ms.G", "tau_ms.B", "z_cam.T",  "z_v_ss.T", "z_v_ms.T"])
    target = os.path.join(target, str(uuid.uuid4()) + ".exr")
    bmp.write(target)

#return a dictionary of numpy array
def bitmap2array(bmp : mi.Bitmap) -> dict:
    data = dict(bmp.split())
    for k, v in data.items():
        data[k] = np.array(v)
        #expand array dimension if not 3
        if(data[k].ndim == 2):
            data[k] = np.expand_dims(data[k], axis=2)
    return data

#remove unnessesary channels
def remove_channels(data, channels):
    for c in channels:
        if c in data:
            del data[c]
        else:
            print("Channel {} not found in data!".format(c))

def pre_process(data: dict):
    #calculate the "gradient" of feature

    #concatenate single-scattering feature
    data['ss'] = np.concatenate((data['Lss'],
                           data['albedo_ss'],
                           data['sigma_s_ss'],
                           data['sigma_t_ss'],
                           data['position_ss'],
                           data['T_ss'],
                           data['tau_ss'],
                           data['z_v_ss'],
                           data['z_cam']), axis=3)
    #concatenate multiple-scattering feature
    data['ms'] = np.concatenate((data['L_ms'],
                           data['albedo_ms'],
                           data['sigma_s_ms'],
                           data['sigma_t_ms'],
                           data['position_ms'],
                           data['T_ms'],
                           data['tau_ms'],
                           data['z_v_ms']), axis=3)
    #remove unnecessary feature
    remove_channels(data, ('Lss', 'albedo_ss', 'sigma_s_ss', 'sigma_t_ss', 'position_ss',
                         'T_ss', 'tau_ss', 'z_v_ss', 'L_ms', 'albedo_ms', 'sigma_s_ms',
                        'sigma_t_ms', 'position_ms', 'T_ms', 'tau_ms', 'z_v_ms', 'z_cam'))
    return data

def getVarianceMap(data, patch_size, relative=False):

  # introduce a dummy third dimension if needed
    if data.ndim < 3:
        data = data[:,:,np.newaxis]

  # compute variance
    mean = ndimage.uniform_filter(data, size=(patch_size, patch_size, 1))
    sqrmean = ndimage.uniform_filter(data**2, size=(patch_size, patch_size, 1))
    variance = np.maximum(sqrmean - mean**2, 0)

    # convert to relative variance if requested
    if relative:
        variance = variance/np.maximum(mean**2, 1e-2)

    # take the max variance along the three channels, gamma correct it to get a
    # less peaky map, and normalize it to the range [0,1]
    variance = variance.max(axis=2)
    variance = np.minimum(variance**(1.0/2.2), 1.0)

    return variance/variance.max()

# Generate importance sampling map based on buffer and desired metric
def getImportanceMap(buffers, metrics, weights, patch_size):
    if len(metrics) != len(buffers):
        metrics = [metrics[0]]*len(buffers)
    if len(weights) != len(buffers):
        weights = [weights[0]]*len(buffers)
    impMap = None
    for buf, metric, weight in zip(buffers, metrics, weights):
        if metric == 'uniform':
            cur = np.ones(buf.shape[:2], dtype=np.float)
        elif metric == 'variance':
            cur = getVarianceMap(buf, patch_size, relative=False)
        elif metric == 'relvar':
            cur = getVarianceMap(buf, patch_size, relative=True)
        else:
            print('Unexpected metric:', metric)
        if impMap is None:
            impMap = cur*weight
        else:
            impMap += cur*weight
    return impMap / impMap.max()

def samplePatchesProg(img_dim, patch_size, n_samples, maxiter=5000):

    # Sample patches using dart throwing (works well for sparse/non-overlapping patches)

    # estimate each sample patch area
    full_area = float(img_dim[0]*img_dim[1])
    sample_area = full_area/n_samples

    # get corresponding dart throwing radius
    radius = np.sqrt(sample_area/np.pi)
    minsqrdist = (2*radius)**2

    # compute the distance to the closest patch
    def get_sqrdist(x, y, patches):
        if len(patches) == 0:
            return np.infty
        dist = patches - [x, y]
        return np.sum(dist**2, axis=1).min()

    # perform dart throwing, progressively reducing the radius
    rate = 0.96
    patches = np.zeros((n_samples,2), dtype=int)
    xmin, xmax = 0, img_dim[1] - patch_size[1] - 1
    ymin, ymax = 0, img_dim[0] - patch_size[0] - 1
    for patch in range(n_samples):
        done = False
        while not done:
            for i in range(maxiter):
                x = randint(xmin, xmax)
                y = randint(ymin, ymax)
                sqrdist = get_sqrdist(x, y, patches[:patch,:])
                if sqrdist > minsqrdist:
                    patches[patch,:] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                minsqrdist = (2*radius)**2

    return patches
 
def prunePatches(shape, patches, patchsize, imp):

    pruned = np.empty_like(patches)

    # Generate a set of regions tiling the image using snake ordering.
    def get_regions_list(shape, step):
        regions = []
        for y in range(0, shape[0], step):
            if y//step % 2 == 0:
                xrange = range(0, shape[1], step)
            else:
                xrange = reversed(range(0, shape[1], step))
            for x in xrange:
                regions.append((x, x + step, y, y + step))
        return regions

    # Split 'patches' in current and remaining sets, where 'cur' holds the
    # patches in the requested region, and 'rem' holds the remaining patches.
    def split_patches(patches, region):
        cur = np.empty_like(patches)
        rem = np.empty_like(patches)
        ccount, rcount = 0, 0
        for i in range(patches.shape[0]):
            x, y = patches[i,0], patches[i,1]
            if region[0] <= x < region[1] and region[2] <= y < region[3]:
                cur[ccount,:] = [x,y]
                ccount += 1
            else:
                rem[rcount,:] = [x,y]
                rcount += 1
        return cur[:ccount,:], rem[:rcount,:]

    # Process all patches, region by region, pruning them randomly according to
    # their importance value, ie. patches with low importance have a higher
    # chance of getting pruned. To offset the impact of the binary pruning
    # decision, we propagate the discretization error and take it into account
    # when pruning.
    rem = np.copy(patches)
    count, error = 0, 0
    for region in get_regions_list(shape, 4*patchsize):
        cur, rem = split_patches(rem, region)
        for i in range(cur.shape[0]):
            x, y = cur[i,0], cur[i,1]
            if imp[y,x] - error > random.random():
                pruned[count,:] = [x, y]
                count += 1
                error += 1 - imp[y,x]
            else:
                error += 0 - imp[y,x]

    return pruned[:count,:]
  
  
def importanceSampling(data, debug=False):
    global patch_size, n_patches

    # extract buffers
    buffers = []
    for b in ['GT', 'color']:
        buffers.append(data[b][:,:,:3])

    # build the metric map
    metrics = ['relvar', 'variance']
    weights = [1.0, 1.0]
    imp = getImportanceMap(buffers, metrics, weights, patch_size)

    if debug:
        print("Importance map:")
        plt.figure(figsize = (15,15))
        imgplot = plt.imshow(imp)
        imgplot.axes.get_xaxis().set_visible(False)
        imgplot.axes.get_yaxis().set_visible(False)
        plt.show()

    # get patches
    patches = samplePatchesProg(buffers[0].shape[:2], (patch_size, patch_size), n_patches)

    if debug:
        print("Patches:")
        plt.figure(figsize=(15, 10))
        plt.scatter(list(a[0] for a in patches), list(a[1] for a in patches))
        plt.show()

    selection = buffers[0] * 0.1
    for i in range(patches.shape[0]):
        x, y = patches[i,0], patches[i,1]
        selection[y:y+patch_size,x:x+patch_size,:] = buffers[0][y:y+patch_size,x:x+patch_size,:]


        
    # prune patches
    pad = patch_size // 2
    pruned = np.maximum(0, prunePatches(buffers[0].shape[:2], patches + pad, patch_size, imp) - pad)
    selection = buffers[0]*0.1
    for i in range(pruned.shape[0]):
        x, y = pruned[i,0], pruned[i,1]
        selection[y:y+patch_size,x:x+patch_size,:] = buffers[0][y:y+patch_size,x:x+patch_size,:]

    if debug:
        print("After pruning:")
        plt.figure(figsize=(15, 10))
        plt.scatter(list(a[0] for a in pruned), list(a[1] for a in pruned))
        plt.show()
        
    return (pruned + pad)

# crops all channels
def crop(data, pos, patch_size):
    half_patch = patch_size // 2
    sx, sy = half_patch, half_patch
    px, py = pos
    return {key: val[(py-sy):(py+sy+1),(px-sx):(px+sx+1),:] for key, val in data.items()}

#return a list of dict
def crop_list(data, debug=False):
    patches = importanceSampling(data, debug)
    return list(crop(data, tuple(pos), patch_size) for pos in patches)


def create_dataset(source, target):
    for file in os.listdir(source):
        if file.endswith(".xml"):
            print(file)
            _data = read_and_render(os.path.join(source , file))
            data = bitmap2array(_data)
            crops = crop_list(data)
            for crop in crops:
                write_exr(crop, target)

if __name__ == "__main__":
    create_dataset("./mitsuba-scene", "./data/train")
    create_dataset("./mitsuba-scene", "./data/test")

