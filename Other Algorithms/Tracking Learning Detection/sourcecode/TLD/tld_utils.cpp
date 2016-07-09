#include "tld_utils.h"

using namespace cv;
using namespace std;

float median(vector<float> v)
{
    //kofpk
    //    int n = floor(v.size() / 2);
    int n = floor(v.size() / 2.0);
    ///kofpk
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

vector<int> index_shuffle(int begin,int end){
    vector<int> indexes(end-begin);
    for (int i=begin;i<end;i++){
        indexes[i]=i;
    }
    random_shuffle(indexes.begin(),indexes.end());
    return indexes;
}

