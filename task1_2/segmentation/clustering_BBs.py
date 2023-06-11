from sklearn.cluster import KMeans
from sklearn import preprocessing as pre
from sklearn.metrics import silhouette_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import cv2

def cluster_bounding_boxes(img, BBs, Centers, num_peaks, k_range, plot=False):
    Y_centers = Centers[:,1]

    Y_in = np.array(Y_centers).reshape(-1, 1)
    scaler = pre.MinMaxScaler()
    Y_in = scaler.fit_transform(Y_in)

    # find the optimal number of clusters (using silhouette score)
    # ideally, 1 line = 1 cluster
    k = find_optimal_k_means(Y_in, start=num_peaks, end=num_peaks+k_range)

    model = KMeans(n_clusters=k)
    preds = model.fit_predict(Y_in)

    centers = scaler.inverse_transform(model.cluster_centers_)
    centers = [int(cent[0]) for cent in centers]
    sorted_centers, sorted_cluster_datapoints = sort_clusters_vertically(k, BBs, preds, centers)

    drawing_img = None
    # If you want to see the plots (they do look nice)
    if plot:
        COLOURS = np.random.randint(0, 255, [k, 3])
        imWidth = img.shape[1]
        drawing_img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        # draw each bounding box, and the center point that was used in clustering
        for idx, (cluster_position, cluster_BBs) in enumerate(zip(sorted_centers, sorted_cluster_datapoints)):
            colour = COLOURS[idx]
            colour = (int(colour[0]), int(colour[1]), int(colour[2]))
            cv2.line(drawing_img, (0, cluster_position), (imWidth, cluster_position), color = colour, thickness=10)

            for x1,y1,x2,y2 in (cluster_BBs):
                colour = (int(colour[0]), int(colour[1]), int(colour[2]))
                cv2.rectangle(drawing_img, (x1,y1), (x2,y2), color = colour, thickness=2)

        for cnt in Centers:  cv2.circle(drawing_img, (cnt[0], cnt[1]), 5, (0,0,255), 5) # red, image is in BGR format
    
    return sorted_cluster_datapoints, drawing_img



def find_optimal_k_means(data, start=2, end=20):
    inertia = []
    silhouette_scores = []

    for k in range(start,end):
        # make and fit the K-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        
        # calculate inertia
        MSD = kmeans.inertia_
        RMSD = np.sqrt(MSD)
        inertia.append(RMSD)

        # calculate silhouette score
        sil_score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(sil_score)

    # the best value for K is the one with the highest silhouette score
    optimal = silhouette_scores.index(max(silhouette_scores))  +  start  # 'index' method starts at 0
    return optimal




# function to re-arrange the clusters so that the list of rows runs from the top
# of the image to the bottom 
def sort_clusters_vertically(k, BBs, preds, centers):
    cluster_datapoints = [[] for i in range(k)]
    
    for idx, bb in enumerate(BBs):
        label = preds[idx]
        cluster_datapoints[label].append(bb)

    sorted_centers, sorted_cluster_datapoints = zip(*sorted(zip(centers, cluster_datapoints)))

    return sorted_centers, sorted_cluster_datapoints



# sorts the bounding boxs of each row (1 row = 1 item in BB_groups = datapoints of 1 cluster0
def sort_BB_clusters_horizontally(BB_groups, right_to_left = True):
    BB_groups_sorted = []
    for group in BB_groups:
        group_srtd = sorted(group, key=lambda x: x[0], reverse=True)
        BB_groups_sorted.append(group_srtd)
    return BB_groups_sorted