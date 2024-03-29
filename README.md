# Vehicle Detection using Machine-Learning

Abstract:

The goal is to design Machine Learning model to identify vehicles in an image. The model is trained with different poses with environmental lighting condition. Vehicle images are collected from the Grupo de Tratamiento de Images (GTI) vehicle image dataset. Different image processing techniques are applied to extract features from the images. Color histogram technique is used to extract color from the image and HOG is to extract HOG features from their corresponding Region of Interests (ROIs). 

The images were trained and tested by Support Vector Machine (SVM), Decision Tree, and Random Forest classifiers using the Scikit-learn in Python. The SVM classifier shows the highest accuracy of 97.8% and the least time taken for training. The model’s performance was evaluated by its accuracy, precision, recall, and F1-score performance metrics. A Sliding window implementation technique is used to slide over an image for searching vehicles in an image and use the SVM classifier to predict vehicles with bounding boxes surrounding it.
