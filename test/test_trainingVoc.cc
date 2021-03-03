/**
 * @file test_traingVoc.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 0.1
 * @date 2021-02-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "frame.h"

std::vector<std::string> loadImages(const std::string &strIndexFilename);

int main(int argc, char** argv)
{
    std::cout << "Reading images ... " << std::endl;
    //Frame::Ptr load_image = std::make_shared<Frame>();
    std::vector<std::string> imageIndex;
    imageIndex = loadImages("/home/ipsg/dataset_temp/image_save/rgb.txt");
    std::vector<cv::Mat> images;
    for(int i=0; i<imageIndex.size(); i++){
        std::cout << "test imageIndex: " << imageIndex[i] << std::endl;
        std::string path = "/home/ipsg/dataset_temp/image_save/"+imageIndex[i];
        // std::string path = "/home/ipsg/dataset_temp/images/" + std::to_string(i+1) + ".png";
        images.push_back(cv::imread(path));
    }
    std::cout << "Images loaded! " << std::endl;
    
    std::cout << "detecting ORB features ... " << std::endl;
    std::vector<std::vector<cv::Mat>> features;
    features.clear();
    features.reserve(images.size());
    for (int i = 0; i < images.size(); i++)
    {
        std::vector<cv::Mat> descriptors;
        // 输入量内参的空构造函数，不使用内参，只对图像进行特征点提取及描述子计算
        Frame::Ptr image_detect = std::make_shared<Frame>(images[i]);
        descriptors = image_detect->toDescriptorVector();
        features.push_back(descriptors);
    }
    std::cout << " features extract done ! " << std::endl;

    // 创建词典，定义词典信息
    // 分支数量10，深度为5
    const int k = 10;
    const int levels = 5;
    const DBoW2::WeightingType weight = DBoW2::TF_IDF;
    const DBoW2::ScoringType score = DBoW2::L1_NORM;

    ORBVocabulary voc(k, levels, weight,score);
    std::string vocName = "/home/ipsg/crane_localization/Vocabulary/MyVoc.txt";
    Frame::createVocabulary(voc,vocName,features);


    return 0;
}

std::vector<std::string> loadImages(const std::string &strIndexFilename)
{
    std::ifstream fIndex;
    std::vector<std::string> imageIndex;
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
        imageIndex.push_back(sRGB);
        }
    }
    return imageIndex;
}
