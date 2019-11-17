import numpy as np
import random as rd
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as sk

filenames = ['animals', 'countries', 'fruits', 'veggies']
category = [0, 1, 2, 3]
wholeData = []
iterations = 50
k_list = list(range(1, 11))
_cluster_of_each_data_point = []
_centroids = []
_count_cluster = []


# Load Data and shuffle
def loadData(flag):
    for itr in range(len(filenames)):
        data_temp = np.genfromtxt(filenames[itr], delimiter=" ")[:, 1:]
        for jtr in range(len(data_temp)):
            if flag == 1:
                data_point = list(data_temp[jtr] / np.linalg.norm(data_temp[jtr]))
            else:
                data_point = list(data_temp[jtr])
            data_point.append(category[itr])
            wholeData.append(data_point)
    rd.shuffle(wholeData)


def randomCentroids(k):
    centroids = []
    centroids_temp = rd.sample(wholeData, k)
    for centroid in centroids_temp:
        centroids.append(centroid[:-1])
    return centroids


def calculateDistance(data_point, centroid, question):
    distance = 0
    if question == 2 or question == 3:
        for i in range(len(data_point)):
            distance += pow((data_point[i] - centroid[i]), 2)
    elif question == 4:
        for i in range(len(data_point)):
            distance += abs(data_point[i] - centroid[i])
    elif question == 5:
        distance_temp = sk.cosine_similarity([data_point], [centroid])
        distance=distance_temp[0][0]
    return distance


def assignClusters(centroids, question):
    cluster_of_each_data_point = [0] * len(wholeData)
    for i in range(len(wholeData)):
        distance_from_centroids = []
        for j in range(len(centroids)):
            distance_jth_centroid = calculateDistance(wholeData[i][:-1], centroids[j],question)
            distance_from_centroids.append(distance_jth_centroid)

        # find min distance index as it is the cluster
        min_distance = min(distance_from_centroids)
        min_distance_centroid = distance_from_centroids.index(min_distance)

        cluster_of_each_data_point[i] = min_distance_centroid
    return cluster_of_each_data_point


def countDatapoints(centroid_index, cluster_of_each_data_point):
    count = 0
    for i in range(len(cluster_of_each_data_point)):
        if cluster_of_each_data_point[i] == centroid_index:
            count += 1
    return count


def makeCluster(index, cluster_of_each_data_point):
    data_of_cluster = []
    for i in range(len(cluster_of_each_data_point)):
        if index == cluster_of_each_data_point[i]:
            data_of_cluster.append(wholeData[i][:-1])
    return data_of_cluster


def calculateNewCentroid(cluster_of_i):
    centroid = []
    for feature_no in range(len(cluster_of_i[0])):
        feature_with_values = []
        for i in range(len(cluster_of_i)):
            feature_with_values.append(cluster_of_i[i][feature_no])

        feature_mean = sum(feature_with_values) / len(feature_with_values)
        centroid.append(feature_mean)
    return centroid


def updateCentroids(cluster_of_each_data_point, centroids, k):
    new_centroids = []

    for i in range(k):
        # Make cluster
        cluster_of_i = makeCluster(i, cluster_of_each_data_point)

        # Calculate new centroid
        len_cluster = len(cluster_of_i)
        if len_cluster != 0:
            centroid = calculateNewCentroid(cluster_of_i)
        else:
            centroid = centroids[i]
        # append to the centroids list
        new_centroids.append(centroid)
    return new_centroids


def compareCentroids(old_centroids, centroids):
    if (old_centroids == centroids):
        return True
    else:
        return False


# Reference : https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
def createConfusionMatrix(cluster_of_each_data_point, centroids, count_cluster):
    confusion_matrix = []  # [TP,FP,FN,TN]
    total_pairs = (len(wholeData) * (len(wholeData) - 1)) / 2

    # Step 1 : Make a matrix whose rows are cluster i.e. k and columns are fixed to 4 according to classes
    matrix_of_cluster_classes = []
    for i in range(len(centroids)):
        centroid_classes_count = []
        count_A = 0
        count_C = 0
        count_F = 0
        count_V = 0
        for j in range(len(cluster_of_each_data_point)):
            if cluster_of_each_data_point[j] == i:
                if wholeData[j][-1] == category[0]:
                    count_A += 1
                elif wholeData[j][-1] == category[1]:
                    count_C += 1
                elif wholeData[j][-1] == category[2]:
                    count_F += 1
                elif wholeData[j][-1] == category[3]:
                    count_V += 1
        centroid_classes_count.append(count_A)
        centroid_classes_count.append(count_C)
        centroid_classes_count.append(count_F)
        centroid_classes_count.append(count_V)

        matrix_of_cluster_classes.append(centroid_classes_count)
    print(matrix_of_cluster_classes)

    # Step 2 Calculate total TP+FP
    TPandFP = 0
    for i in range(len(centroids)):
        TPandFP += (count_cluster[i] * (count_cluster[i] - 1)) / 2

    # Step 3 Calculate TP
    TP = 0
    for i in range(len(matrix_of_cluster_classes)):
        for j in range(len(matrix_of_cluster_classes[i])):
            TP += (matrix_of_cluster_classes[i][j] * (matrix_of_cluster_classes[i][j] - 1)) / 2

    confusion_matrix.append(TP)

    # Step 4 Calculate FP
    FP = TPandFP - TP
    confusion_matrix.append(FP)

    # Step 5 Calculate total FNandTN
    FNandTN = total_pairs - TPandFP

    # Step 5 Calculate FN
    FN = 0
    for i in range(len(category)):
        col_values = []
        FN_col = 1
        for j in range(len(matrix_of_cluster_classes)):
            col_values.append(matrix_of_cluster_classes[j][i])

        for k in range(len(col_values)):
            if col_values[k] != 0:
                FN_col *= col_values[k]
        FN += FN_col

    confusion_matrix.append(FN)

    # Step 6 Calculate TN
    TN = FNandTN - FN
    confusion_matrix.append(TN)

    print(confusion_matrix)

    return confusion_matrix


def calculatePrecision(confusion_matrix):
    if confusion_matrix[0] != 0:
        precision = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1])
        return precision
    elif confusion_matrix[0] == 0:
        return 0
    elif confusion_matrix[0] + confusion_matrix[1] == 0:
        return 0


def calculateRecall(confusion_matrix):
    if confusion_matrix[0] != 0:
        recall = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[2])
        return recall
    elif confusion_matrix[0] == 0:
        return 0
    elif confusion_matrix[0] + confusion_matrix[2] == 0:
        return 0


def calculateFscore(precision, recall):
    if precision + recall == 0:
        return 0
    elif precision == 0 or recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)


def plotGraph(precision_list, recall_list, fscore_list, flag):
    plt.plot(k_list, precision_list, '-x', label="Precision")
    plt.plot(k_list, recall_list, '-o', label="Recall")
    plt.plot(k_list, fscore_list, '-D', label="F Score")
    plt.xlim(0, len(k_list) + 1)
    plt.ylim(0, 1)
    plt.xlabel("K values")
    plt.ylabel(" Range")
    plt.legend()
    plt.show()

def clearGlobals():
    global wholeData
    global _cluster_of_each_data_point
    global _count_cluster
    global _centroids

    wholeData=[]
    _cluster_of_each_data_point=[]
    _count_cluster=[]
    _centroids=[]


# ---------------------------K Means -----------------------------------
def Kmeans(flag, question):
    loadData(flag)

    recall_list = []
    precision_list = []
    fscore_list = []
    global _cluster_of_each_data_point
    global _count_cluster
    global _centroids

    for _k in k_list:

        print("--------------", _k, "-------------------")
        # Step 1 : Get random centroids for the dataset
        print("Initial centroids")
        _centroids = randomCentroids(_k)
        # print(_centroids)

        _exitcode = 0

        # Step 2 : Loop over cluster making with the help of distance function calculation and updating centroid
        for _itr2 in range(0, iterations):
            print("Entering iteration no-", _itr2)
            _exitcode = _itr2

            # Step 2.1 Assign datapoints to nearest centorid and make clusters
            _cluster_of_each_data_point = assignClusters(_centroids, question)

            # count how many data points belong to same centroid
            _count_cluster = [0] * len(_centroids)
            for _itr3 in range(len(_centroids)):
                _count_cluster[_itr3] = countDatapoints(_itr3, _cluster_of_each_data_point)
            print("Number of data points related to each cluster")
            print(_count_cluster)

            # Step 2.2 Update centroids of the every clusters
            _old_centroids = _centroids.copy()
            _centroids = updateCentroids(_cluster_of_each_data_point, _centroids, _k)
            # print(_centroids)

            print("Exiting iteration no-", _exitcode)

            # Step 2.3 Check if old and new centroids are same , if same then break the loop as found solution
            if compareCentroids(_old_centroids, _centroids):
                print("break called after iteration no-", _exitcode)
                break

        _confusion_matrix = createConfusionMatrix(_cluster_of_each_data_point, _centroids, _count_cluster)

        # Precision,recall and fscore calulation and append to respective list
        _precision = calculatePrecision(_confusion_matrix)
        precision_list.append(_precision)
        _recall = calculateRecall(_confusion_matrix)
        recall_list.append(_recall)
        fscore_list.append(calculateFscore(_precision, _recall))

    print("K values", k_list)
    print("Precision", precision_list)
    print("Recall", recall_list)
    print("Fscore", fscore_list)

    plotGraph(precision_list, recall_list, fscore_list, 0)

    clearGlobals()
