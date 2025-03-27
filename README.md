# Dataset

The dataset used is ShanghaiTech dataset available here : [Drive Link](https://drive.google.com/file/d/1xnMNCrxTwNc_oeQhmzRNUbzZb-dmauuZ/view?usp=drive_link)

The dataset is divided into two parts, A and B. Part A consists of images with a high density of crowd. Part B consists of images with images of sparse crowd scenes.

## Data Preprocessing

In data preprocessing, the main objective was to convert the ground truth provided by the ShanghaiTech dataset into density maps. For a given image the dataset provided a sparse matrix consisting of the head annotations in that image. This sparse matrix was converted into a 2D density map by passing through a Gaussian Filter. The sum of all the cells in the density map results in the actual count of people in that particular image. Refer the `Preprocess.ipynb` notebook for the same.
