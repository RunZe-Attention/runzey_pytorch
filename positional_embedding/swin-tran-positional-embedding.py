import torch

def get_2D_relative_position_index(height,width):
    coords = torch.stack(torch.meshgrid(torch.arange(height),torch.arange(width)))
    coords_flatten = torch.flatten(coords,1)
    print(coords_flatten)

    print(coords_flatten[:,:,None])
    print(coords_flatten[:,None,:])
    relative_coords_bias = coords_flatten[:,:,None] - coords_flatten[:,None,:]
    relative_coords_bias[0, :, :] += height-1
    relative_coords_bias[1, :, :] += width - 1

    relative_coords_bias[0,:,:] *= relative_coords_bias[1,:,:].max()+1
    return relative_coords_bias.sum(0)




if __name__ == '__main__':
    get_2D_relative_position_index(2,2)
