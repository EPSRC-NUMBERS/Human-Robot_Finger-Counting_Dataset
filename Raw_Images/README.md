# Folders with raw images.

## Description

This section contains the raw images of hands performing finger-counting from 1 to 5.

### A few notes on the structure:
1. The images are divided into two types: iCub robot and human.
2. Finger-counting on human hands was done on two separate occasions, therefore giving two sets.
3. Finger-counting with iCub was done once, but pictured with iCub's left and right cameras. Only one hand (left) was used due to technical issues.
This gives two sets with similar, but slightly different images.
4. For human hands the number of images in each class (`1`-`5`) is variable and includes different ways one can show a given number.
For iCub, numbers were given using the standard of the American Sign Language.
5. For iCub, each class has 200±2 images.
6. Class `0` contains empty background for that particular setting.
7. For the robot hands the sets are copied into three folders:
  * `Stream` contains all the images,
  * `Full` contains images grouped into classes,
  * `Cleaned` folder is basically `Full` with the most ambiguous images removed (subjective!).
