# THOMAS: Thalamus-Optimized Multi-Atlas Segmentation
Segmentation of the thalamus into 12 nuclei using the white-matter-nulled image contrast and PICSL's joint label fusion.  Note that this requires priors/ provided elsewhere.

Please see the new version of this project [here](https://github.com/thalamicseg/thomas_new), which implements STTHOMAS and provides priors/.

## Requirements
- [ANTs](https://github.com/stnava/ANTs.git)
	- NOTE: before April 2016, there was a bug in ANTs/Examples/antsJointFusion.cxx when using the "-x" switch, please rebuild ANTs from the latest source to fix this.
- [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
- [convert3d](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D)

## Installation
- git clone https://github.com/thalamicseg/thomas.git
- Extract THOMAS-priors.zip to thomas/
- python require.py

## Usage
- python THOMAS_v0.py -h
- Example: python THOMAS_v0.py -p 4 --tempdir ants --jointfusion image.nii.gz ./ ALL
	- tempdir is often useful in case something goes wrong, you can resume from previous attempts.
	- jointfusion calls the original implementation of the [PICSL MALF algorithm](https://www.nitrc.org/projects/picsl_malf) instead of antsJointFusion.  This was used in the publication.
- swapdimlike.py - reorients an image to match the orientation of another
- form_multiatlas.py - combines many independent labels together into a single atlas
