import os

import jax
import jax.numpy as jnp
import numpy as np
import argparse
import cv2

from jaxtyping import Float

from dlt import render
from dlt import primary_sample_space as pss
from dlt import experiment_utils

from mcmc.chainstate import load_chain_state

def render_lens(lens_array : Float, outname, focal_length, imres, rng_seed=0, sensor_height_ratio=1.0, offset=(0.0, 0.0)):
    spp = 800
    full_sensor_height = 36.0
    sens_height = full_sensor_height * sensor_height_ratio
    imheight = full_sensor_height / jnp.sqrt((full_sensor_height/2)**2 + focal_length**2)
    nbatches = imres * imres * spp // (imres * spp)
    offset_vec = jnp.array([0.0, -offset[0] * full_sensor_height/2, -offset[1] * full_sensor_height/2])

    # usaf_sampler = render.image_sampler(None, imheight, imheight)
    # cone_angle ignored by farfield rendering
    usaf_sampler = render.image_sampler('data/images/USAF-1951.png', imheight, imheight)
    sens = render.QuantizedRect(sens_height, imres, sens_height, imres, 10.0, jnp.pi/8)
    
    rng_key = jax.random.PRNGKey(rng_seed)
    im, valid_ratio  = render.render_zemax_image_farfield(lens_array, sens, usaf_sampler, spp=spp, rng_key=rng_key, nbatches=nbatches, verbose=True, offset=offset_vec, color=True)

    # normalize the image for sensor size
    im = im / (sensor_height_ratio * sensor_height_ratio)

    print('Valid ratio: ', valid_ratio)
    if valid_ratio < 0.01:
        print('Warning: too many rays are invalid. Valid ratio: ', valid_ratio)

    cleaned_outname = os.path.splitext(outname)[0]
    jnp.save(cleaned_outname + '.npy', im)

    print('Max value: ', im.max())
    # im_np = np.array(im)
    # cv2.imwrite(cleaned_outname + '.png', (255 * im_np / im_np.max()).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lensfile', type=str, default='results/addrem_50mm/best_lenses/restore.npz')
    parser.add_argument('--outname', type=str, default='results/images/test.png')
    parser.add_argument('--rng_seed', type=int, default=0)
    parser.add_argument('--sens_height', type=float, default=1.0)
    parser.add_argument('--offset_x', type=float, default=0.0)
    parser.add_argument('--offset_y', type=float, default=0.0)
    parser.add_argument('--focal_length', type=float, default=50.0)
    parser.add_argument('--imres', type=int, default=256)
    parser.add_argument('--filetype', type=str, default='chain')

    jax.config.update("jax_enable_x64", True)

    args = parser.parse_args()
    print(args)

    os.makedirs(os.path.dirname(args.outname), exist_ok=True)

    if args.filetype == 'chain':
        chain_state = load_chain_state(args.lensfile)
        lens = chain_state.cur_state
        reglens = pss.normalized2lens(lens)
        lens_array = reglens.toarray()
    elif args.filetype == 'zemax':
        # lens_array = zemax_loader.load_zemax_file(args.lensfile)
        reglens = experiment_utils.load_lens_file(args.lensfile, 10)
        lens_array = reglens.toarray()
    else:
        raise ValueError('Unknown file type: ', args.filetype)


    render_lens(lens_array, 
                args.outname, 
                args.focal_length, 
                args.imres, 
                args.rng_seed, 
                args.sens_height,
                (args.offset_x, args.offset_y))
