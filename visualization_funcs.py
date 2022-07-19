import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Normalization function for centering to 0 the attribution map visualization
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# Clip highest or lowest values in the top percentile
def clip(mapping,percentile=99.99):
    or_shape = mapping.shape
    mapping_max = np.percentile(np.abs(mapping.flatten()),percentile)
    mapping = mapping.clip(-mapping_max,mapping_max)
    mapping = mapping.reshape(or_shape)
    return mapping

# Normalize tensor to values between 0 and 1
def normalize_tensor(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / (tensor.max()+1e-8)
    return tensor

# Plot tensor as image
def show_tensor(tensor, title='',ax=None,cmap=None , normalize=True):
    if(normalize):
        tensor = normalize_tensor(tensor)
    if(cmap is None):
        cmap =  'viridis'
    if(ax is None):
        plt.imshow(tensor,cmap=cmap)
        plt.axis('off')
        plt.title(title, fontsize=9)
    else:
        ax.imshow(tensor,cmap=cmap)
        ax.axis('off')
        ax.set_title(title, fontsize=9)

# Normalize positive and negative attributions in the range between -1,1 and
# return the attribution map
def normalize_attributions(pos_attribution,neg_attribution):
    # Clip the lowest or largest values for better visualization
    pos_attribution = clip(pos_attribution)
    neg_attribution = clip(neg_attribution)
    # Set min value to 0
    pos_attribution = pos_attribution-pos_attribution.min()
    neg_attribution = neg_attribution-neg_attribution.min()

    # Get highest or lowest value
    max_pos_neg_attribution = max(np.abs(pos_attribution).max(),
                               np.abs(neg_attribution).max())
    pos_attribution = pos_attribution/max_pos_neg_attribution
    neg_attribution = neg_attribution/max_pos_neg_attribution

    return pos_attribution-neg_attribution

def normalize_uncertainty(uncertaintymap,percentile):
    # Clip the lowest or largest values for better visualization
    uncertaintymap = clip(uncertaintymap,percentile)
    # Set min value to 0
    uncertaintymap = uncertaintymap-uncertaintymap.min()

    # Get highest or lowest value
    max_uncertaintymap = np.abs(uncertaintymap).max()
    uncertaintymap = uncertaintymap/max_uncertaintymap

    return uncertaintymap


# Visualize DMBP results (attribution maps and linear mappings)
def visualize_attributions(results, target_class='',
                           imfile=None):
    num_models = len(results.keys())
    figure, axes = plt.subplots(1,4)

    ## Show Original Image
    image = results['image']
    ax = axes[0]
    show_tensor(image,ax=ax,title='Class: ' + target_class)


    ## Show linear mappings
    pos_mapping = results['pos_mapping']
    neg_mapping = results['neg_mapping']

    # Clip the lowest or largest values of the linear mapping for better visualization
    pos_neg_mappings = np.concatenate((pos_mapping,neg_mapping),axis=2)
    pos_neg_mappings = clip(pos_neg_mappings)
    pos_mapping = pos_neg_mappings[:,:,0:3]
    neg_mapping = pos_neg_mappings[:,:,3::]
    # Show
    ax = axes[2]
    show_tensor(pos_mapping,ax=ax,title='Pos. Mapping')
    ax = axes[3]
    show_tensor(-neg_mapping,ax=ax,title='Neg. Mapping')

    ## Show Attribution Map
    # Combine positive and negative attributions
    pos_attribution = results['pos_attribution'].mean(axis=2)
    neg_attribution = -results['neg_attribution'].mean(axis=2)
    attributions = normalize_attributions(pos_attribution,neg_attribution)

    # Show
    ax = axes[1]
    ax.imshow(normalize_tensor(image))
    ax.imshow(attributions,cmap='seismic',clim=(-1,1),norm=MidpointNormalize(midpoint=0,vmin=-1, vmax=1),alpha=0.7)
    ax.axis('off')
    ax.set_title('Attribution Map',fontsize=9)

    # Show figure or save into a file
    if(imfile is None):
        plt.show()
    else:
        plt.subplots_adjust(hspace = 0.01, wspace=0.01,left=0, bottom=0, right=1, top=1)
        #plt.tight_layout()
        plt.savefig(imfile,dpi=600,bbox_inches='tight')


# Visualize our results (uncertainty maps and linear mappings)
def visualize_uncertainty(image, pred, mse, uncertaintymap, savefolder=None, img_index=None):
    '''
    params:
    - results: dict, including 'image', 'mse', 'uncertaintymap'
    '''
    figure = plt.figure()

    ## Show Original Image
    # image = results['image']
    # ax = axes[0]
    # show_tensor(image,ax=ax,title='Class: ' + target_class)

    ## Show Uncertainty Map
    uncertaintymap = normalize_uncertainty(uncertaintymap,percentile=100)
    # uncertaintymap /= np.max(uncertaintymap)
    msemap = normalize_uncertainty(mse,percentile=98.5)

    # Show
    plt.imshow(normalize_tensor(pred)) # (0~1)
    # plt.imshow(uncertaintymap,cmap='seismic',clim=(-1,1),norm=MidpointNormalize(midpoint=0,vmin=-1, vmax=1),alpha=0.6)
    plt.imshow(uncertaintymap,cmap='seismic',clim=(-1,1),alpha=0.6)
    plt.axis('off')
    plt.savefig(savefolder+'/{:02d}_vis_std.png'.format(img_index),bbox_inches = 'tight',pad_inches = 0)

    # mse + pred
    plt.imshow(normalize_tensor(pred)) # (0~1)
    plt.imshow(msemap,cmap='seismic',clim=(-1,1),norm=MidpointNormalize(midpoint=0,vmin=-1, vmax=1),alpha=0.5)
    # plt.imshow(msemap,cmap='seismic',clim=(-1,1),alpha=0.7)
    plt.axis('off')
    plt.savefig(savefolder+'/{:02d}_vis_mse.png'.format(img_index),bbox_inches = 'tight',pad_inches = 0)
    plt.close()

    # # Show figure or save into a file
    # if(imfile is None):
    #     plt.show()
    # else:
    #     plt.subplots_adjust(hspace = 0.01, wspace=0.01,left=0, bottom=0, right=1, top=1)
    #     #plt.tight_layout()
    #     plt.savefig(imfile,dpi=600,bbox_inches='tight')

 
# Visualize our results (uncertainty maps and linear mappings)
def visualize_depthmap(pred, uncertaintymap, normalize_divider, savefolder=None, img_index=None):
    '''
    params:
    - results: dict, including 'image', 'mse', 'uncertaintymap'
    '''
    ## Show Original Image
    # image = results['image']
    # ax = axes[0]
    # show_tensor(image,ax=ax,title='Class: ' + target_class)

    ## Show Uncertainty Map
    uncertaintymap = uncertaintymap / normalize_divider 
    uncertaintymap_zero = uncertaintymap.clip(0,0.001)

    figure = plt.figure()
    plt.imshow(pred,cmap='gray') # (0~1)
    plt.imshow(uncertaintymap_zero,cmap='seismic',clim=(-1,1),norm=MidpointNormalize(midpoint=0,vmin=-1, vmax=1),alpha=0.4)
    # plt.imshow(uncertaintymap,cmap='seismic',clim=(-1,1),alpha=0.7)
    plt.axis('off')
    plt.savefig(savefolder+'/vis_depth_{:02d}.png'.format(img_index),bbox_inches = 'tight',pad_inches = 0)
    plt.close()

    # Show
    figure = plt.figure()
    plt.imshow(pred,cmap='gray') # (0~1)
    plt.imshow(uncertaintymap,cmap='seismic',clim=(-1,1),norm=MidpointNormalize(midpoint=0,vmin=-1, vmax=1),alpha=0.4)
    # plt.imshow(uncertaintymap,cmap='seismic',clim=(-1,1),alpha=0.7)
    plt.axis('off')
    plt.savefig(savefolder+'/vis_depth_std_{:02d}.png'.format(img_index),bbox_inches = 'tight',pad_inches = 0)
    plt.close()

    return figure

def visualize_rgbmap(pred, uncertaintymap, normalize_divider, savefolder=None, img_index=None):
    '''
    params:
    - results: dict, including 'image', 'mse', 'uncertaintymap'
    '''
    ## Show Original Image
    # image = results['image']
    # ax = axes[0]
    # show_tensor(image,ax=ax,title='Class: ' + target_class)

    ## Show Uncertainty Map
    uncertaintymap = np.minimum(uncertaintymap,normalize_divider)
    uncertaintymap = uncertaintymap-uncertaintymap.min()
    uncertaintymap = uncertaintymap / normalize_divider
    # uncertaintymap = normalize_uncertainty(uncertaintymap,percentile=99.9)

    # Show
    figure = plt.figure()
    plt.imshow(normalize_tensor(pred)) # (0~1) 
    plt.imshow(uncertaintymap,cmap='seismic',clim=(-1,1),norm=MidpointNormalize(midpoint=0,vmin=-1, vmax=1),alpha=0.7)
    # plt.imshow(uncertaintymap,cmap='seismic',clim=(-1,1),alpha=0.7)
    plt.axis('off')
    plt.savefig(savefolder+'/vis_rgb_std_{:02d}.png'.format(img_index),bbox_inches = 'tight',pad_inches = 0)
    plt.close()

    return figure
