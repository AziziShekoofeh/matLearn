# matLearn
matLearn: machine learning algorithm implementations in Matlab 


## Disclaimer: 

This package is the result of a group effort during our Machine Learning course in the Fall 2014, CPSC 540 taught by Dr. Mark Schmid at UBC. Back then, it wasn't any Tensorflow, Caffe, Torch, and PyTorch library to make your life this much easy, and most of the machine learning and deep learning implementations have been done using MATLAB. I clearly remember to write a Deep Belief Network (DBN) library (+3K line of codes) in MATLAB for my first *deep learning* paper in 2014 when the first versin of Caffe and TF had been released early 2015. So, I think keep tracking the history is sometimes useful, and maybe in future somebody needs this package! 

You can find the full description and the info about main people behind collecting, testing, and updating the package in the [project website](https://www.cs.ubc.ca/~schmidtm/Software/matLearn.html). 


## Description 
The matLearn package contains Matlab implementations of a wide variety of the most commonly-used machine learning algorithms, all using a simple common interface. It in particular focuses on the following tasks:

 - **Regression**: Predict a continuous output variable given observed (continuous or discrete) features. It includes methods that are robust, non-parametric, kernelized, and/or ordinal methods.
 
- **Classification**: Predict a discrete class label given observed (continuous or discrete) features. It includes methods for binary, multiclass, ordinal, and multi-label settings.

- **Clustering**: Group together similar unlabelled data points. It includes parametric and non-parametric methods.

- **Dimensionality Reduction**: Learn a low-dimensional representation of high-dimensional data while trying to maintain the structure in the higher dimensions.

- **Density Estimation**: Construct an estimate of an underlying probability density function using observations in the data matrix.


 ###  Authors
The matLearn package contains code from the following sources:

The 2014 version of matLearn consisted of code by writing by the students of the the Fall 2014 section of CPSC 540 taught by Mark Schmidt at UBC. Individual contributors to the final package were: 

Adrian Wong | Alim Virani | Alireza Shafaei | Antoine Ponsard | Anurag Ranjan | Ben Bougher | Ben Zhu | Bita Nejat | Celia Siu | Daniel Fugere | Delaram Behnami | Fujun Xie | Giorgio Gori | Giovanni Vivian | Issam Laradji | Jason Hartford | Jeff Allen | João Cardoso | Kamyar Ardekani | Ken Lau | Keyulu Xu | Manyou Ma | Matthew Dirks | Nasim (Sedigheh) Zolaktaf | Nathaniel Lim | Nazanin Hamzei | Neil Newman | Neil Traft | Philipp Witte | Radhika Nangia | Rebecca McKnight | Rindra Ramamonjison | Roee Bar | Sampoorna Biswas | Scott Sallinen | Seyed Ali Saberali | Sharan Vaswani | Shekoofeh Azizi | Xi Laura Cang | Yan Peng | Yan Zhao

Jennifer (Xin Bei) She (contact: x.she@alumni.ubc.ca) organized the previous version of matLearn into a useable form in summer 2015, and added a variety of new methods to the package. A poster prepared by Jennifer is available here.

Geoffrey Roeder (contact: geoff.roeder@gmail.com) merged the existing matLearn library with Mark Schmidt's implementations (dating back to 2004) in summer in 2016, and added a variety of new methods/tasks including unsupervised learning.


### Citations

If you use this software in a publication, please cite the version of the work based on the year the package was downloaded using the following information:

- G. Roeder, X. She, M. Schmidt, et al. matLearn: machine learning algorithm implementations in Matlab , 2016.

-  X. She, M. Schmidt, et al. matLearn: machine learning algorithm implementations in Matlab , 2015.
