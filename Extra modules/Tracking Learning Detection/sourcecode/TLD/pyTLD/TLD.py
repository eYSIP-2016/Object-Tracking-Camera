import FerNNClassifier
import cv2
import numpy as np
import math
from operator import itemgetter

classifier = FerNNClassifier.FerNNClassifier()
scales = []
grid = []
good_boxes = []
bad_boxes = []
best_box = []

#Bounding Box Parameters
min_win = 15
#Genarator Parameters
#initial parameters for positive examples
patch_size = 15
num_closest_init = 10
num_warps_init = 20
noise_init = 5
angle_init = 20
shift_init = 0.02
scale_init = 0.02
#update parameters for positive examples
num_closest_update = 10
num_warps_update = 10
noise_update = 5
angle_update = 10
shift_update = 0.02
scale_update = 0.02
#parameters for negative examples
bad_overlap = 0.2
bad_patches = 100

bbhull = []

#Classifier Parameters
valid = 0.5
ncc_thesame = 0.95
nstructs = 10
structSize = 13
thr_fern = 0.5
thr_nn = 0.65
thr_nn_valid = 0.7

def init(frame1, box):
    #Get Bounding Boxes
    buildGrid(frame1,box)
    #Preparation
    #allocation
    bbox_step =7
    #Init Generator
    getOverlappingBoxes(box,num_closest_init)
    #Correct Bounding Box
    lastbox=best_box
    lastconf=1
    lastvalid=True
    #Prepare Classifier
    classifier_prepare(scales);
    ///Generate Data
    // Generate positive data
    generatePositiveData(frame1,num_warps_init);
    // Set variance threshold
    Scalar stdev, mean;
    meanStdDev(frame1(best_box),mean,stdev);
    integral(frame1,iisum,iisqsum);
    var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);
    // Generate negative data
    generateNegativeData(frame1);
    //Split Negative Ferns into Training and Testing sets (they are already shuffled)
    int half = (int)nX.size()*0.5f;
    nXT.assign(nX.begin()+half,nX.end());
    nX.resize(half);
    ///Split Negative NN Examples into Training and Testing sets
    half = (int)nEx.size()*0.5f;
    nExT.assign(nEx.begin()+half,nEx.end());
    nEx.resize(half);
    //Merge Negative Data with Positive Data and shuffle it
    vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
    vector<int> idx = index_shuffle(0,ferns_data.size());
    int a=0;
    for (int i=0;i<pX.size();i++){
        ferns_data[idx[a]] = pX[i];
        a++;
    }
    for (int i=0;i<nX.size();i++){
        ferns_data[idx[a]] = nX[i];
        a++;
    }
    //Data already have been shuffled, just putting it in the same vector
    vector<cv::Mat> nn_data(nEx.size()+1);
    nn_data[0] = pEx;
    for (int i=0;i<nEx.size();i++){
        nn_data[i+1]= nEx[i];
    }
    ///Training
    classifier.trainF(ferns_data,2); //bootstrap = 2
    classifier.trainNN(nn_data);
    ///Threshold Evaluation on testing sets
    classifier.evaluateTh(nXT,nExT);
}

def buildGrid(img, box)
    SHIFT = 0.1
    SCALES = np.array[0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                      0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                      2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174]
    #Rect bbox
    bbox = [0,0,0,0,0,0]
    sc=0
    for s in range(21):
        temp = box[2] * SCALES[s]
        if((int(temp * 10)) % 10 < 5):
            width = math.floor(temp)
        else:
            width = math.ceil(temp)
        temp = box[3] * SCALES[s]
        if((int(temp * 10)) % 10 < 5):
            height = math.floor(temp)
        else:
            height = math.ceil(temp)
        min_bb_side = min(height,width)
        rows,cols = img.shape
        if (min_bb_side < min_win || width > cols || height > rows)
            continue;
        scale = [width,height]
        scales.append(scale)
        temp = SHIFT * min_bb_side
        if((int(temp * 10)) % 10 < 5):
            temp = math.floor(temp)
        else:
            temp = math.ceil(temp)
        for y in range(1,rows-height,temp):
            for x in range(1,cols-width,temp):
                bbox[0] = x
                bbox[1] = y
                bbox[2] = width
                bbox[3] = height
                bbox[4] = bbOverlap(x,y,width,height,box)
                bbox[5] = sc
                grid.append(bbox)
        sc=sc+1

def bbOverlap(x,y,width,height,box2)
    if (x > box2[0]+box2[2]):
        return 0.0
    if (y > box2[1]+box2[3]):
        return 0.0
    if (x+width < box2[0]):
        return 0.0
    if (y+height < box2[1]):
        return 0.0

    colInt =  min(x+width,box2[0]+box2[2]) - max(x, box2[0])
    rowInt =  min(y+height,box2[1]+box2[3]) - max(y,box2[1])

    intersection = colInt * rowInt
    area1 = width*height
    area2 = box2[0]*box2[1]
    return intersection / (area1 + area2 - intersection)

def getOverlappingBoxes(box1,num_closest):
    max_overlap = 0.0
    for i in range(len(grid)):
        if (grid[i][4] > max_overlap):
            max_overlap = grid[i][4]
            best_box = grid[i]
        if (grid[i][4] > 0.6):
            good_boxes.append(i)
        else if (grid[i][4] < bad_overlap):
            bad_boxes.append(i)
    #Get the best num_closest (10) boxes and puts them in good_boxes
    sorted(good_boxes, key=itemgetter(4))
    good_boxes = good_boxes[0:num_closest]
                    
    getBBHull()

def getBBHull(self):
    x1=65000
    x2=0
    y1=65000
    y2=0
    for i in range(len(good_boxes)):
        idx= good_boxes[i]
        x1=min(grid[idx][0],x1)
        y1=min(grid[idx][1],y1)
        x2=max(grid[idx][0]+grid[idx][2],x2)
        y2=max(grid[idx][1]+grid[idx][3],y2)
    bbhull = [x1,y1,x2-x1,y2 -y1]

def classifier_prepare(scales)
    acum = 0
    #Initialize test locations for features
    totalFeatures = nstructs*structSize
    features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
    RNG& rng = theRNG();
    float x1f,x2f,y1f,y2f;
    int x1, x2, y1, y2;
    for (int i=0;i<totalFeatures;i++){
        x1f = (float)rng;
        y1f = (float)rng;
        x2f = (float)rng;
        y2f = (float)rng;
        for (int s=0;s<scales.size();s++){
            x1 = x1f * scales[s].width;
            y1 = y1f * scales[s].height;
            x2 = x2f * scales[s].width;
            y2 = y2f * scales[s].height;
            features[s][i] = Feature(x1, y1, x2, y2);
        }

    }
    //Thresholds
    thrN = 0.5*nstructs;

    //Initialize Posteriors
    for (int i = 0; i<nstructs; i++) {
        posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
        pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
        nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
    }
}
    
