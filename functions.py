import numpy as np
import matplotlib.pyplot as plt
import random
import os
import shutil
import tempfile
import git

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def loadDataset():
    if not os.path.isdir('./data'):
        print('la cartella ./data non è presente !')
        temp = tempfile.mkdtemp()
        git.Repo.clone_from('https://github.com/zalandoresearch/fashion-mnist.git', temp, branch='master', depth=1)
        print('clonazione effettuata')
        os.mkdir('./data')
        if not os.path.isdir('./data/fashion'):
            print('copio il dataset')
            shutil.move(os.path.join(temp, 'data/fashion'), './data')
        if not os.path.isdir('./data/mnist_reader.py'):
            print('copio il parser')
            shutil.move(os.path.join(temp, 'utils/mnist_reader.py'), './data')

    from data.mnist_reader import load_mnist

    XTrain, yTrain = load_mnist('data/fashion', kind='train')
    XTest, yTest = load_mnist('data/fashion', kind='t10k')

    return XTrain, yTrain, XTest, yTest


def scaleData(trainX, testX):
    sc = StandardScaler()
    sc.fit(trainX)
    trainXStd = sc.transform(trainX)
    testXStd = sc.transform(testX)
    return trainXStd, testXStd


def showImage(i, images, trueLabels, pred=None, group=False):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = images[i]

    if pred is None:  # stampo un'immagine specifica con la sua label
        label = trueLabels[i]
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel("{}".format(class_names[label]), color='green')

    else:  # stampo un'immagine specifica con la label predetta dal classificatore
        predLab = pred[i]
        trueLab = trueLabels[i]
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        print("predLab: ", predLab)

        if predLab == trueLab:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {} ({})".format(class_names[predLab],
                                             predLab, class_names[trueLab]), color=color)

    if group is False:
        plt.show()


def plotImages(rows, cols, images, trueLabels, pred):
    num_rows = rows
    num_cols = cols
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        showImage(i, images, trueLabels, pred=pred, group=True)

    plt.tight_layout()
    plt.show()


def plotLearningCurve(estimator, trainX, trainY, testX, testY, iter):

    trainSizes = [10, 10000, 20000, 30000, 40000, 50000, 60000]

    trainIndexes = list(range(len(trainX)))  # 60.000 indici -- da 0 a 59999 -- ad ogni immagine associo un indice numerico  -- in ordine 'cronologico'

    testIndexes = list(range(len(testX)))  # 10.000 indici -- da 0 a 9999 -- ad ogni label associo un indice numerico -- in ordine 'cronologico'

    # Matrici per gli score e per le predizioni

    scoresTr = np.zeros((iter, len(trainSizes)))
    scoresTe = np.zeros((iter, len(trainSizes)))

    for i in range(iter):
        print("\nIterazione numero: ", i)
        # print('contenuto trainX: ', trainX[:3, :15])

        # Shuffle del training set

        random.shuffle(trainIndexes)
        currentTrainIndexes = trainIndexes

        random.shuffle(testIndexes)
        currentTestIndexes = testIndexes

        currentXTrain = trainX[currentTrainIndexes]
        print("Numero di righe currentXTrain: ", len(currentXTrain))
        print('Numero di colonne currentXTrain: ', len(currentXTrain[0]))
        currentYTrain = trainY[currentTrainIndexes]
        currentXTest = testX[currentTestIndexes]
        currentYTest = testY[currentTestIndexes]

        print("contenuto currentXTrain: ", currentXTrain[:3, :15])

        # Ciclo su trainSizes

        for j in range(len(trainSizes)):
            s = trainSizes[j]
            print("\nDimensione corrente: ", s)
            cXtrain = currentXTrain[:s]
            cyTrain = currentYTrain[:s]
            # print('X: ', cXtrain)
            # print('Y: ', cyTrain)
            print("Numero di righe cXtrain: ", len(cXtrain))
            print('Numero di colonne cXtrain: ', len(cXtrain[0]))
            print("Numero di elementi in cyTrain: ", len(cyTrain))

            # Fit

            estimator.fit(cXtrain, cyTrain)

            # Training accuracy

            yPredTrain = estimator.predict(cXtrain)
            trainScore = accuracy_score(yPredTrain, cyTrain)
            print("Training accuracy: ", trainScore)
            scoresTr[i][j] = trainScore

            # Testing accuracy

            yPred = estimator.predict(currentXTest)
            # plotImages(2, 3, currentXTest, currentYTest, pred=yPred)
            print("PREDIZIONEEEEE: ", yPred)
            print("LUNGHEZZA PREDIZIONE: ", len(yPred))
            testScore = accuracy_score(yPred, currentYTest)
            print("Testing accuracy: ", testScore)
            scoresTe[i][j] = testScore

    print(scoresTe)
    print('Righe matrice degli score per il test', len(scoresTe), '\nColonne matrice degli score per il test: ', len(scoresTe[0]))

    # Media e deviazione standard dei training e dei testing scores raccolti, per ogni train size

    trainMeanScores = np.mean(scoresTr, axis=0)
    print("trainMeanScores: ", trainMeanScores)
    traindStdScores = np.std(scoresTr, axis=0)

    testMeanScores = np.mean(scoresTe, axis=0)
    print("testMeanScores: ", testMeanScores)
    testdStdScores = np.std(scoresTe, axis=0)

    # Plot delle curve di apprendimento

    plt.fill_between(trainSizes, trainMeanScores - traindStdScores, trainMeanScores + traindStdScores, alpha=0.1, color='orange')
    plt.fill_between(trainSizes, testMeanScores - testdStdScores, testMeanScores + testdStdScores, alpha=0.1, color='green')

    plt.plot(trainSizes, trainMeanScores, 'o-', color='orange', label='Training score')
    plt.plot(trainSizes, testMeanScores, 'o-', color='green', label='Test score')

    plt.ylabel('Accuracy Score')
    plt.xlabel('Training Set Sizes')
    plt.legend(loc='best')
    plt.title('Learning Curves')
    plt.grid()
    plt.show()






