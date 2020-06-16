# Folders with raw images.

## Description

This section contains the raw images of hands performing finger-counting from 1 to 5.

### A few notes on the structure:
1. The images are divided into two types: iCub robot and human.
2. Finger-counting on human hands was done on two separate occasions, therefore two sets are present.
3. Finger-counting with iCub was done once, but pictures are taken separately with iCub's left and right cameras. Only one hand (left) was used due to technical issues. This gives two sets with similar images, but taken from a slightly different perspective.
4. Class `0` contains empty background for that particular setting.
5. For human hands, the number of images in each class (`1` to `5`) is variable and, occasionally, includes different ways one can show a given number.
6. For iCub, numbers were given using the standard of the American Sign Language, and each class has 200±2 images. The sets are organised into three folders:
  * `Stream` contains all the images,
  * `Full` contains images grouped into classes,
  * `Cleaned` folder contains the images from the `Full` folder, minus the images that look most ambiguous to interpret (subjective!).
