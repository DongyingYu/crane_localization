/**
 * @file localization.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 0.1
 * @date 2021-02-28
 *
 * @copyright Copyright (c) 2021
 */

#include "localization.h"

Localization::Localization(int cnt) : win_size_(cnt)
{
    loadVocabulary();
    processKeyimages();
    vscore_.reserve(frames_.size());
    for(int i=0; i<frames_.size();i++)
    {
        vscore_.push_back(std::make_pair(0.0,0));
        // vscore_.resize(frames_.size());
    }
}

Localization::~Localization()
{
}

int Localization::videoCapturescore(const cv::Mat &video_image/*const std::string &image_address*/)
{
    // cv::Mat image = cv::imread(image_address);
    cv::Mat image = video_image.clone();
    Frame::Ptr image_temp = std::make_shared<Frame>(image, pVocabulary_);
    image_temp->computeBoW();
    DBoW2::BowVector v2 = image_temp->bow_vec_;

    std::vector<float> score_temp;
    for (int i = 0; i < frames_.size(); i++)
    {
        DBoW2::BowVector v1 = frames_[i]->bow_vec_;
        float score = image_temp->computeScore(v1,v2);
        //std::cout << "image " << i << "  compare with the input frame " << " the score is : " << score << std::endl;

        if(score < 0 || score >1)
            score = 0.0;
        score_temp.push_back(score); 
        //std::cout << "The size of sore_temp is : " << score_temp.size() << std::endl;
    }

    if(winFrames_.size() >= win_size_) {
        winFrames_.pop_front();
    }    
    winFrames_.push_back(score_temp);

    for (int i = 0; i < score_temp.size(); i++)
    {
        float sum_score_temp = 0.0;
        for (int j = 0; j < winFrames_.size(); j++)
        {
            sum_score_temp = sum_score_temp + winFrames_[j][i];
        }
        vscore_[i].first = sum_score_temp;
        // std::cout << "the for circle number: " << i << std::endl;
        vscore_[i].second = i;

        // std::cout << "The test index of frame : " << vscore_[i].second << std::endl;
    }
    // 降序排列，筛选出排在前五的的分值
    // first_flag_ = false;
    sort(vscore_.begin(),vscore_.end(),compareScore);
    std::cout << "The size of vscore_: " << vscore_.size() << std::endl;
    for (int i = 0; i < 5; i++)
    {
        std::cout << "The " << i << "  number of vscore_ is : " << vscore_[i].first << std::endl;
    }
    // 计算共享单词个数
    // for ( DBoW2::BowVector::const_iterator vit=frames[frames.size()-1]->bow_vec_.begin(),
    //       vend=frames[frames.size()-1]->bow_vec_.end();vit != vend; vit++)
    // {
    //     std::cout << " the id of BowVoc: " << vit->first << std::endl;
    // }

    std::cout << " The max score is : " << vscore_[0].first << std::endl;
    std::cout << " The index number of image is :  " << vscore_[0].second << std::endl;

    const int height = max(frames_[vscore_[0].second]->img_.rows, image.rows);
    const int width = frames_[vscore_[0].second]->img_.cols + image.cols;
    cv::Mat output(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    frames_[vscore_[0].second]->img_.copyTo(output(cv::Rect(0, 0, frames_[vscore_[0].second]->img_.cols, frames_[vscore_[0].second]->img_.rows)));
    image.copyTo(output(cv::Rect(frames_[vscore_[0].second]->img_.cols, 0, image.cols, image.rows)));

    cv::resize(output, output, {0, 0}, 0.6, 0.6);
    cv::imshow("Image contrast", output);
    return vscore_[0].second; 
}

void Localization::loadVocabulary()
{
    std::cout << std::endl << "[INFO]: Loading Vocabulary ..." << std::endl;
    pVocabulary_ = new ORBVocabulary();
    bool bVocLoad = pVocabulary_->loadFromTextFile("/home/ipsg/crane_localization/Vocabulary/ORBvoc.txt");

    if(!bVocLoad)
    {
        std::cout << "[error]: Wrong path to vocabulary. " << std::endl;
        std::cout << "[error]: Falied to open at the given path. " << std::endl;
        exit(-1);
    }
    std::cout << "Vocabulary loaded! " << std::endl;
}

void Localization::loadImages(const std::string &strIndexFilename)
{
    std::ifstream fIndex;
    fIndex.open(strIndexFilename.c_str());
    while (!fIndex.eof())
    {
        std::string s;
        getline(fIndex,s);
        if(!s.empty()){
        std::stringstream ss;
        ss << s;
        // double cnt;
        std::string sRGB;
        ss >> sRGB;
        // ss >> cnt;
        vimageIndex_.push_back(sRGB);
        }
    }
}

void Localization::processKeyimages()
{
    loadImages("/home/ipsg/dataset_temp/image_save/rgb.txt");
    std::vector<cv::Mat> images;
    std::cout << "The number of images:  " << vimageIndex_.size() << std::endl;
    for(int i=0; i<vimageIndex_.size(); i++){
        std::cout << "test imageIndex: " << vimageIndex_[i] << std::endl;
        std::string path = "/home/ipsg/dataset_temp/image_save/"+vimageIndex_[i];
        std::cout << "images path: " << path << std::endl;
        images.push_back(cv::imread(path));
    }

    std::cout << "detecting ORB features ... " << std::endl;
    for (int i = 0; i < images.size(); i++)
    {
        // 对图像进行特征点提取及描述子计算
        Frame::Ptr image_detect = std::make_shared<Frame>(images[i], pVocabulary_);
        image_detect->computeBoW();
        frames_.emplace_back(image_detect);
    }
}

// 设定从大到小排序
bool Localization::compareScore(const pair<float,int> a,const pair<float,int> b){
    return a.first > b.first;
}
