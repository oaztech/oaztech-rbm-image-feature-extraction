import pickle

import numpy as np
from skimage.transform import resize


def extractFeatureImage(image_arr):
    image_resized = resize(image_arr, (77, 65), mode='constant', anti_aliasing=False)
    image_flattened = image_resized.flatten('C')  # flatten img type C
    image_subclassed = np.asarray(image_flattened, 'float32')  # subclassing to float

    # load trained modal
    with open('assets/trained_modals/saved_modal.pickle', 'rb') as handle:
        rbm_features_classifier = pickle.load(handle)

    return rbm_features_classifier.predict([image_subclassed])[0]  # predicting class


def training_modal():
    import pickle

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import convolve
    from skimage.io import imread_collection, imshow
    from skimage.transform import resize
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline

    # Read in all centered images, and examine the size of the first image
    imgs = imread_collection('assets/centered/*.gif')
    print("Imported", len(imgs), "images")
    print("The first one is", len(imgs[0]), "pixels tall, and",
          len(imgs[0][0]), "pixels wide")

    # Let's confirm that they're all the same size:
    print(np.std([len(x) for x in imgs]))
    print(np.std([len(x[0]) for x in imgs]))

    # # Examples of Images

    # Example of Happy
    imshow(imgs[13])

    # Example of Glasses
    imshow(imgs[23])

    # Example of shading
    imshow(imgs[113])

    # Let's resize; at this 231x165, the RBM may have memory issues.
    # this reduces our image vectors from length of 38,115 (231x165)
    # to length of 5,005 (77x65), significantly improving performance

    imgs = [resize(x, (77, 65), mode='constant', anti_aliasing=False) for x in imgs]

    # Show an example of a reduced image
    imshow(imgs[113])

    # Flatten all images to arrays
    imgsarr = [x.flatten('C') for x in imgs]

    # # Goal
    # Use RBM to perform feature extraction on these images, and examine the features.

    # Create a target variable: 1 through 15 for each of the 15 subjects
    Y = [[_ for i in range(1, 12)] for _ in range(1, 16)]
    Y = [num for sub in Y for num in sub]

    # Define the RBM, used for feature generation
    rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=.01,
                       n_iter=20, n_components=150)

    # Define the Classifier - Logistic Regression will be used in this case
    logistic = LogisticRegression(solver='lbfgs', max_iter=10000,
                                  C=6000, multi_class='multinomial')

    # Combine the two into a Pipeline
    rbm_features_classifier = Pipeline(
        steps=[('rbm', rbm), ('logistic', logistic)])

    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(imgsarr, Y)

    # Confirm success by predicting the training data
    Y_pred = rbm_features_classifier.predict(imgsarr)
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(Y, Y_pred)))

    # Let's view the 150 components created by the RBM

    plt.figure(figsize=(15, 15))
    for i, comp in enumerate(rbm.components_[:150]):
        plt.subplot(15, 10, i + 1)
        plt.imshow(comp.reshape((77, 65)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('150 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()

    # Selected features for closer examination
    first = 8
    second = 45

    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(rbm.components_[first].reshape((77, 65)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.subplot(122)
    plt.imshow(rbm.components_[second].reshape((77, 65)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # # Let's try adding 5x more data by creating copies of these images shifted a pixel in each direction
    # Function credit to [scikit-learn](https://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-auto-examples-neural-networks-plot-rbm-logistic-classification-py)

    def nudge_dataset(X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 77x65 images in X around by 1px to left, right, down, up
        Credit to SciKitLearn (linked above)
        """
        direction_vectors = [
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]]

        def shift(x, w):
            return convolve(x.reshape((77, 65)), mode='constant', weights=w).ravel()

        X = np.concatenate([X] +
                           [np.apply_along_axis(shift, 1, X, vector)
                            for vector in direction_vectors])
        Y = np.concatenate([Y for _ in range(5)], axis=0)
        return X, Y

    X = np.asarray(imgsarr, 'float32')
    Xbig, Ybig = nudge_dataset(X, Y)

    # Redefine the RBM
    rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=.01,
                       n_iter=20, n_components=150)

    # Redefine the Logistic Classifier
    logistic = LogisticRegression(solver='lbfgs', max_iter=10000,
                                  C=6000, multi_class='multinomial')

    # Combine the two into a Pipeline
    rbm_features_classifier = Pipeline(
        steps=[('rbm', rbm), ('logistic', logistic)])

    rbm_features_classifier.fit(Xbig, Ybig)

    Y_pred = rbm_features_classifier.predict(Xbig)
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(Ybig, Y_pred)))

    # ## With the Added Data and 150 components, this pipeline has predicted the training data perfectly.
    # Note that it's probably overfit and this is not ideal for new data, but it is for our goal of examining the components a RBM can extract.

    # Components for examination
    plt.figure(figsize=(15, 15))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(15, 10, i + 1)
        plt.imshow(comp.reshape((77, 65)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('150 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()

    # Selected features for closer examination
    toshow = [104, 116, 84]

    plt.figure(figsize=(16, 10))
    for i, comp in enumerate(toshow):
        plt.subplot(1, 3, i + 1)
        plt.imshow(rbm.components_[comp].reshape((77, 65)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()

    # # Analysis
    #
    # For some hidden components, like the left and center examples above, it's intuitive to see how they are storing information about image content, and how they contribute to recreations of the original (during fitting) or predictions. Others, like the right, are less so. This is not an indication of usefuleness; even if a hidden component is not intuitively explaining patterns to the human eye, it can combine with other hidden components to make accurate predictions.

    # save training
    with open('assets/trained_modals/saved_modal.pickle', 'wb') as handle:
        pickle.dump(rbm_features_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
