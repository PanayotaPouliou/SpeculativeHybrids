import pathlib
from pathlib import Path
import json
from pyexpat import model
import numpy as np
from sklearn.cluster import KMeans

root = Path(__file__).absolute().parent.parent
config_file = root / "dataset_config.json"

def read_pointclouds(points_path,semantic_path,sample_path):
    """
        args:
            points_path: pathlib.PATH for pointclouds
            semantic_path: pathlib.PATH for semantic labels
            sample_path: pathlib.PATH for pointcloud sample

        return:
            points: [n,3] sampled points 
            sem_labels: [n] sampled labels
    """
    points = np.loadtxt(points_path).astype(np.float32)
    sample = np.loadtxt(sample_path).astype(np.int64)
    sem_label = np.loadtxt(semantic_path).astype(np.int64)

    points = points[sample]
    sem_label = sem_label[sample]

    return points,sem_label

def get_sem_counts(sem_label):
    """
        args:
            sem_label: [n] semantic label
        
        return:
            classes: INT semantic label classes count
            counts: [k] points num for every of k semantic label
    """
    counts = []
    for item in sem_label:
        while item > len(counts):
            counts.append(0)
        counts[item - 1] += 1
    return len(counts),counts

def get_sem_split(sem_counts,points_num,k):
    """
        args:
            sem_counts: [k] points num for every of k semantic label
            points_num: INT number of points in pointcloud
            k: INT number of sampled points
        
        output:
            dvi: [k] sampled points num for every semantic label     
    """
    ratio = [sem_counts[i] / points_num * k for i in range(len(sem_counts))]
    ratio_int_part = [int(ratio[i]) for i in range(len(ratio))]
    ratio_float_part = [(ratio[i] - ratio_int_part[i],i) for i in range(len(ratio))]
    dvi = ratio_int_part
    cnt = 0
    for item in ratio_int_part:
        cnt += item
    rest = k - cnt
    
    ratio_float_part.sort(key=lambda x : -x[0])
    for i in range(rest):
        dvi[ratio_float_part[i][1]] += 1
    
    return dvi

def k_means(points,k,id):
    """
    args:
        points: [n] point cloud of same semantic label
        k: INT the k in k_means
        id: INT id of the semantic label
    
    return:
        sampled_points: [k,4] center of k clusters
    """
    estimator = KMeans(n_clusters=k)
    estimator.fit(points)
    centroids = estimator.cluster_centers_
    ids = [[id] for i in range(k)]
    print (centroids)
    return np.concatenate([centroids, ids],axis=1)

from GNG_implementation.neuralgas import GrowingNeuralGas

def gng(points,k,id):

    arr = np.array(points)
    print (np.shape(arr))

    """
    args:
        points: [n] point cloud of same semantic label
        k: INT the k in k_means -> n_inputs
        id: INT id of the semantic label
    
    return:
        sampled_points: [k,4] center of k clusters
    """

    gng = GrowingNeuralGas(
        max_neurons = k, 
        max_iter = 100,
        max_age= 10,
        eb = 0.1,
        en = 0.006,
        alpha = 0.5,
        beta = 0.995,
        l =1,
        dataset = arr
    )

    centroids, model = gng.learn()
    centroids = np.array(centroids)
    centroids = centroids[:-1]

    print (k)
    print(np.shape(centroids))
    
    ids = [[id] for i in range(k)]
    ids = np.array(ids)
    print(np.shape(ids))
    print(type(ids))

    return np.concatenate([centroids, ids],axis=1)


def get_avg_sem_samples(points,sem_label,k):
    """
    args: 
        points: [n,3] pointclouds
        sem_label: [n] semantic label
        k: INT number of sampled points we want to get
    
    return:
        sem_classes: INT number of semantic classes
        sampled_points: [k * 4] k points from sampled results concat with its semantic label
    """
    sem_classes,counts = get_sem_counts(sem_label)
    points_num = len(points) #2048?

    sub_points = [[] for i in range(sem_classes)] #In our case sub_points=[[],[],[],[],[]]
    sem_split = get_sem_split(counts,points_num,k) #This is a list with the number of points ??that all have to sum at 32?? on each label ex. 1 -> 17pnts 2->5 3->5 3-> 100 4-> 4 5-> 1 

    for i in range(points_num):
        sem = sem_label[i] # =1 =wall
        sub_points[sem - 1].append(points[i]) # appends all the points of the same label in an array. all arrays from all labels are in one array called sub_points
    
    sampled_points = None
    for i in range(sem_classes):
        if sem_split[i] == 0:
            continue
            
        if sampled_points is None:
            sampled_points = gng(sub_points[i],sem_split[i],i + 1) #chanche with gng
        else:
            sampled_points = np.concatenate(
                [sampled_points,
                gng(sub_points[i],sem_split[i],i + 1)], #chanche with gng
                axis = 0
            )
    return sem_classes,sampled_points

def sample_semantic_points(config,k = 32):
    """
    args:
        config: {} config dict
        k: INT number of sampled points
        type: "avg" or "equal" or "nosem"
    
    return:
        None
    """
    cat = {}

    cat_path = root / config["category_information_path"]
    with cat_path.open("r") as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]    
    
    meta = {}
    for item in cat:
        meta[item] = []
        dir_point = root / Path(config["full_pointcloud_path"] ) / cat[item] 
        dir_seg = root / Path(config["semantic_label_path"] ) / cat[item]
        dir_sampled = root / Path(config["pointcloud_sample_path"] ) / cat[item]
        dir_sem_sampling = None
        dir_sem_sampling = root / Path(config["semantic_avg_sampled_pointcloud_path"]) / cat[item]
        
        fns = dir_point.iterdir()

        for fn in fns:
            token = fn.stem
            meta[item].append((
                dir_point / (token + ".pts"),
                dir_seg / (token + ".seg"),
                dir_sampled / (token + ".sam"),
                dir_sem_sampling / (token + ".pts"),
                dir_sem_sampling
            ))
    
    meta_info = {}
    for cls_key in meta:
        print(cls_key)
        cls_list = meta[cls_key]
        max_sem_classes = 0
        cnt = 0

        for item in cls_list:
            if cnt % 100 == 0:
                print(cnt)
            cnt += 1
            if not item[4].exists():
                item[4].mkdir(parents=True,exist_ok=True)
            out_fp = item[3].open('w')

            points,sems = read_pointclouds(item[0],item[1],item[2]) #points_path,semantic_path,sample_path

            sem_classes,out_pointclouds = get_avg_sem_samples(points,sems,k)
            
            max_sem_classes = max(sem_classes,max_sem_classes)
            np.savetxt(
                out_fp,
                out_pointclouds
            )
            out_fp.close()
        
        meta_info[cls_key] = {
            "max_sem_classes": max_sem_classes,
            "item_num": len(cls_list)
        }
        print(meta_info[cls_key])

    path_meta = root / Path(config["semantic_avg_sampled_pointcloud_path"]) / "meta.json" 
    meta_fp = path_meta.open('w')
    json.dump(
        meta_info,
        meta_fp
    )
    meta_fp.close()


if __name__ == "__main__":
    with config_file.open("r") as f:
        config = json.load(f)["dataset_config"]
        print("Sampling structure points")
        sample_semantic_points(config)
