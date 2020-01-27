#!/usr/bin/env python
"""
Take a THOMAS output and attempt to segment CL and VPM by warping an atlas to native space and filling in holes in the THOMAS segmentation.


"""

import os
import sys
import argparse
import tempfile
import time
import glob
import numpy as np
import nibabel
import libraries.parallel as parallel
from shutil import rmtree
from functools import partial
from pprint import pprint
from datetime import timedelta
from skimage import morphology
from libraries.imgtools import check_run, flip_lr, ants_compose_a_to_b
from libraries.ants_nonlinear import ants_apply_warp
from THOMAS_constants import template, this_path,  roi_choices


def detect_side():
    pass


def merge_atlas(primary_fname, secondary_fname, output_fnames_dict, background_values={0}, selem=None):
    """
    Merge two atlases together.  The primary atlas is preserved.  New ROIs in the secondary atlas are dilated and used to fill in the "cracks" of the primary atlas.  These should be two NIfTI paths.
    """
    primary = nibabel.load(primary_fname)
    primary.data = primary.get_data().astype(int)
    secondary = nibabel.load(secondary_fname)
    secondary.data = secondary.get_data().astype(int)
    # get all ROI values and remove background ROI
    labels_in_primary = set(np.unique(primary.data)) - background_values
    mask_primary = np.isin(primary.data, list(labels_in_primary))
    # keep only new ROIs, keep only voxels that are not primary atlas or background values
    mask_secondary = np.isin(secondary.data, list(labels_in_primary), invert=True) * np.isin(secondary.data, list(background_values), invert=True)

    new_nuclei = secondary.data * mask_secondary
    new_nuclei_values = set(np.unique(new_nuclei)) - background_values
    new_nuclei = morphology.dilation(new_nuclei, selem=selem)
    # new_nuclei = morphology.closing(new_nuclei, selem=np.ones((3, 3, 3)))
    merged_atlas = new_nuclei * np.invert(mask_primary) + primary.data * mask_primary
    for value in new_nuclei_values:
        nii = nibabel.Nifti1Pair(merged_atlas == value, primary.get_affine(), header=primary.get_header())
        nibabel.save(nii, output_fnames_dict[value])


def find_unique(fname):
    files = glob.glob(fname)
    if len(files) == 1:
        return files.pop()
    else:
        print('%s not found or non-unique`:' % fname)
        print files
        raise AssertionError


parser = argparse.ArgumentParser(description='Segmentation of the CL and VPM thalamic nuclei given the outputs from a THOMAS-based segmentation. Please use ALL ROIs for this to work correctly.')
parser.add_argument('input_image', help='input WMnMPRAGE NiFTI image, may need to be in LR PA IS format')
parser.add_argument('tempdir', help='the temporary directory used in THOMAS to re-use the nonlinear warps.')
parser.add_argument('output_path', help='the output directory for segmented ROIs')
# parser.add_argument('-w', '--warp', metavar='path', help='looks for {path}InverseWarp.nii.gz and {path}Affine.txt instead of basing it off input_image.')
# parser.add_argument('-F', '--forcereg', action='store_true', help='force ANTS registration to WMnMPRAGE mean brain template. The --warp argument can be then used to specify the output path.')
parser.add_argument('-p', '--processes', nargs='?', default=1, const=None, type=int, help='number of parallel processes to use.')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
parser.add_argument('-d', '--debug', action='store_true', help='debug mode, interactive prompts')
parser.add_argument('-R', '--right', action='store_true', help='segment right thalamus')
parser.add_argument('--dilation', default=None, type=int, help='amount of dilation to apply to CL and VPM nuclei to fill in gaps of THOMAS segmentation.  By default a connectivity-1 cross-kernel is used.  Otherwise specify a number to use an NxNxN cube. ')


REQUIRED_ROIS = ['2-AV', '4-VA', '5-VLa', '6-VLP', '7-VPL', '8-Pul', '9-LGN', '10-MGN', '11-CM', '12-MD-Pf', '13-Hb', '14-MTT']
ATLAS = os.path.join(this_path, 'atlas_with_CL_VPM.nii.gz')


def main(args, temp_path, pool):
    t = time.time()
    # Check files
    required_rois = [os.path.join(temp_path, el + '.nii.gz') for el in REQUIRED_ROIS]
    exist_required_rois = map(os.path.exists, required_rois)
    if not all(exist_required_rois):
        print('Some files not found:')
        pprint(dict(zip(required_rois, exist_required_rois)))
        raise AssertionError
    # Form_multiatlas
    thomas_multiatlas = os.path.join(temp_path, 'CLVPM-THOMAS-multiatlas.nii.gz')
    print('--- Joining THOMAS nuclei into a single atlas. --- Elapsed: %s' % timedelta(seconds=time.time() - t))
    check_run(thomas_multiatlas, parallel_command, os.path.join(this_path, 'form_multiatlas.py') + ' Metric %s %s' % (thomas_multiatlas, ' '.join(required_rois)))

    # Find warps
    inverse_warp = find_unique(os.path.join(temp_path, '*InverseWarp.nii.gz'))
    affine = find_unique(os.path.join(temp_path, '*Affine.txt'))

    # Warp atlas to native space
    input_image = ATLAS
    input_affine = '-i %s' % affine
    input_warp = inverse_warp
    atlasCLVPM_in_native = os.path.join(temp_path, 'CLVPM-' + os.path.basename(ATLAS))
    switches = '--use-NN'
    print('--- Warping CL-VPM template atlas to native space. --- Elapsed: %s' % timedelta(seconds=time.time() - t))
    # Note that the argument order for affine and warp below are swapped as we are going from template to native space
    check_run(atlasCLVPM_in_native, ants_apply_warp, args.input_image, input_image, input_affine, input_warp, atlasCLVPM_in_native, switches, execute=parallel_command)

    # Merge atlas
    merged_atlas = {
        17: os.path.join(temp_path, '17-CL.nii.gz'),
        18: os.path.join(temp_path, '18-VPM.nii.gz')
    }
    print('--- Merging THOMAS segmentation and CL-VPM atlas. --- Elapsed: %s' % timedelta(seconds=time.time() - t))
    if args.dilation is None:
        dilation_element = None
    else:
        dilation_element = np.ones(3*(args.dilation,))
    merge_atlas(thomas_multiatlas, atlasCLVPM_in_native, merged_atlas, selem=dilation_element)

    # Resort output to original ordering
    output_path = args.output_path
    files = [(fname, os.path.join(output_path, os.path.basename(fname))) for fname in merged_atlas.values()]
    if args.right:
        pool.map(flip_lr, files)
        files = [(el, el) for _, el in files]
    pool.map(parallel_command, [
        '%s %s %s %s' % (os.path.join(this_path, 'swapdimlike.py'), in_file, args.input_image, out_file) for in_file, out_file in files
    ])
    print '--- Finished --- Elapsed: %s' % timedelta(seconds=time.time() - t)


if __name__ == '__main__':
    args = parser.parse_args()
    exec_options = {'echo': False, 'suppress': False}
    if args.verbose:
        exec_options['verbose'] = True
    if args.debug:
        print 'Debugging mode forces serial execution.'
        args.processes = 1
    parallel_command = partial(parallel.command, **exec_options)
    pool = parallel.BetterPool(args.processes)
    print 'Running with %d processes.' % pool._processes
    # TODO Add path of script to command()
    # os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.dirname(sys.argv[0]))
    if args.tempdir:
        temp_path = args.tempdir
    #     if not os.path.exists(temp_path):
    #         print 'Making %s' % os.path.abspath(temp_path)
    #         os.makedirs(temp_path)
    # else:
    #     temp_path = tempfile.mkdtemp(dir=os.path.dirname(args.output_path))
    try:
        main(args, temp_path, pool)
    finally:
        pool.close()
        # Clean up temp folders
        if not args.debug and not args.tempdir:
            try:
                rmtree(temp_path)
            except OSError as exc:
                if exc.errno != 2:  # Code 2 - no such file or directory
                    raise
