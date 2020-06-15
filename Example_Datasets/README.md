### This is an example workflow on how to create an evenly combined robot-human dataset with 2000 training, 400 testing and 400 validation images (2000-400-400 for short).

## Part I. Sampling
### Step 1.
Using `split_images.py`, create a sample of 100-20-20 images for each class (1 to 5) from the iCub left camera set.  
*Results are in `Robot_Fingers_500_100_100`*
### Step 2.
Using the same procedure, create the same sample range for iCub right camera set.
Apply `mirror_images.py` to the resulting files to get the data for the right hand.  
*Results are in `Robot_Fingers_Mirrored_500_100_100`*
### Step 3.
Again, using the script `split_images.py`, create a sample of 1000-200-200 images for the human hands from `Set1`.
For cases where there are not enough images in the class (namely, `4` and `5`) to fill all three sets, some workarounds exist.
One might either use `Set2` to fill in the gaps in the validation set (since validation set does not require silhouette extraction), or re-use a small number of `Set1` training images in testing or validation sets.  
*Results are in `Human_Fingers_1000_200_200`*
## Part II. Silhouette extraction
### Step 4.
Using the script `hand_silhouetting.py`, extract the hand silhouettes for training and testing for each of three datasets (**6 times** in total).
Template .cfg files are given for both of the hand types.
The script also applies alternative backgrounds, however, this is optional.  
*Results for each dataset are in the `silhouettes` sub-folders.*
## Part III. Dataset merge
### Step 5.
Using the `merge_datasets.py` script, combine all the three datasets into one. For example, `Robot_Fingers_500_100_100` can be merged with `Robot_Fingers_Mirrored_500_100_100` into `Robot_Fingers_1000_200_200` and this resulting dataset can be merged with `Human_Fingers_1000_200_200` into `Combined_Fingers_2000_400_400`. The procedure has to be done separately for training and testing sets. Images from the validation sets can be simply copied into one folder.  
*Results are in the `Combined_Fingers_2000_400_400` and `Robot_Fingers_1000_200_200`.*
