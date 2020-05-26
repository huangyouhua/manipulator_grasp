/***********************************************************************
Copyright 2019 Wuhan PS-Micro Technology Co., Itd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***********************************************************************/

#include "probot_grasping/vision_manager.h"

using namespace cv;
using namespace std;

VisionManager::VisionManager(ros::NodeHandle n_, float length, float breadth) : it_(n_)
{
    this->table_length = length;
    this->table_breadth = breadth;

    image1_pub_ = it_.advertise("/table_detect", 1);
	image2_pub_ = it_.advertise("/object_detect", 1);
}

bool VisionManager::get2DLocation(cv::Mat img, float &x, float &y)
{
    this->curr_img = img;
    img_centre_x_ = img.rows / 2;
    img_centre_y_ = img.cols / 2;

    static bool first_detect = true;

    static cv::Rect tablePos;


    ros::NodeHandle n;
    int object_color;
	bool action_flag = true;

    if (first_detect)
    {
        object_color = 0;
        detectTable(tablePos);
        detect2DObject(x, y, tablePos, object_color);
        first_detect = false;
        action_flag = true;
    }
    else
    {
        if (!n.getParam("graspingDemo/object_color", object_color))
        {
			object_color = 5;
            ROS_INFO_STREAM("Cannot get param graspingDemo/object_color..");
        }

        if (object_color < 0 || object_color > 2)
            action_flag = false;
		else
		{
			action_flag = true;
			detect2DObject(x, y, tablePos, object_color);
		}
			

    }

    convertToMM(x, y);

    return action_flag;
}

void VisionManager::detectTable(cv::Rect &tablePos)
{
    // Extract Table from the image and assign values to pixel_per_mm fields
    cv::Mat BGR[3];
    cv::Mat image = curr_img.clone();
    split(image, BGR);
    cv::Mat gray_image_red = BGR[2];
    cv::Mat gray_image_green = BGR[1];
    cv::Mat denoiseImage;
    cv::medianBlur(gray_image_red, denoiseImage, 3);

    // Threshold the Image
    cv::Mat binaryImage = denoiseImage;
    for (int i = 0; i < binaryImage.rows; i++)
    {
        for (int j = 0; j < binaryImage.cols; j++)
        {
            int editValue = binaryImage.at<uchar>(i, j);
            int editValue2 = gray_image_green.at<uchar>(i, j);

            if ((editValue >= 0) && (editValue < 20) && (editValue2 >= 0) && (editValue2 < 20))
            { // check whether value is within range.
                binaryImage.at<uchar>(i, j) = 255;
            }
            else
            {
                binaryImage.at<uchar>(i, j) = 0;
            }
        }
    }
    dilate(binaryImage, binaryImage, cv::Mat());

    // Get the centroid of the of the blob
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(binaryImage, nonZeroPoints);
    cv::Rect bbox = cv::boundingRect(nonZeroPoints);
    cv::Point pt;
    pt.x = bbox.x + bbox.width / 2;
    pt.y = bbox.y + bbox.height / 2;
    cv::circle(image, pt, 2, cv::Scalar(0, 0, 255), -1, 8);

    // Update pixels_per_mm fields
    pixels_permm_y = bbox.height / table_length;
    pixels_permm_x = bbox.width / table_breadth;

    tablePos = bbox;

    // Test the conversion values
    std::cout << "Pixels in y" << pixels_permm_y << std::endl;
    std::cout << "Pixels in x" << pixels_permm_x << std::endl;

    // Draw Contours - For Debugging
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binaryImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::drawContours(image, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
    }
}

bool VisionManager::detect2DObject(float &pixel_x, float &pixel_y, cv::Rect &tablePos, int object_color)
{
    vector<int> iLowH = {100, 35, 0};
    vector<int> iLowS = {43, 43, 60};
    vector<int> iLowV = {46, 46, 46};

    vector<int> iHighH = {124, 77, 10};
    vector<int> iHighS = {255, 255, 255};
    vector<int> iHighV = {255, 255, 255};

    cv::Mat image = curr_img.clone();
    std::vector<cv::Mat> hsvSplit;
    cv::Mat imgHSV;
    cv::Mat binaryImage;
    cvtColor(image, imgHSV, COLOR_BGR2HSV);
    split(imgHSV, hsvSplit);
    equalizeHist(hsvSplit[2], hsvSplit[2]);
    inRange(imgHSV, Scalar(iLowH[object_color], iLowS[object_color], iLowV[object_color]), Scalar(iHighH[object_color], iHighS[object_color], iHighV[object_color]), binaryImage); //Threshold the image

    blur(binaryImage, binaryImage, Size(3, 3)); //mean value fiter
    dilate(binaryImage, binaryImage, cv::Mat());

    // Get the centroid of the of the blob
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(binaryImage, nonZeroPoints);
    cv::Rect bbox = cv::boundingRect(nonZeroPoints);
    cv::Point pt;
    pixel_x = bbox.x + bbox.width / 2;
    pixel_y = bbox.y + bbox.height / 2;

    // Test the conversion values
    std::cout << "pixel_x" << pixel_x << std::endl;
    std::cout << "pixel_y" << pixel_y << std::endl;

    // For Drawing
    pt.x = bbox.x + bbox.width / 2;
    pt.y = bbox.y + bbox.height / 2;
    cv::circle(image, pt, 2, cv::Scalar(0, 0, 255), -1, 8);

    // Draw Contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binaryImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255, 0, 0);
        cv::drawContours(image, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
    }

    sensor_msgs::ImagePtr object_detect_image = cv_bridge::CvImage(std_msgs::Header(), "mono8", binaryImage).toImageMsg();
    // //sensor_msgs::ImagePtr object_detect_image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    image2_pub_.publish(object_detect_image);

    return true;
}

void VisionManager::convertToMM(float &x, float &y)
{
    // Convert from pixel to world co-ordinates in the camera frame
    x = (x - img_centre_x_) / pixels_permm_x;
    y = (y - img_centre_y_) / pixels_permm_y;
}
