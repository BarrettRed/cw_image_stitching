#include <opencv2/opencv.hpp>
#include <iostream>

double computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat I1, I2;
    img1.convertTo(I1, CV_32F);
    img2.convertTo(I2, CV_32F);

    cv::Mat I1_2 = I1.mul(I1);       
    cv::Mat I2_2 = I2.mul(I2);       
    cv::Mat I1_I2 = I1.mul(I2);      

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);   

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);   

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    
    cv::Scalar mssim = cv::mean(ssim_map);
    return mssim[0]; 
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <reference_image> <panorama_ORB> <panorama_SIFT>\n";
        return -1;
    }

    cv::Mat ref = cv::imread(argv[1]);
    cv::Mat pan1 = cv::imread(argv[2]);
    cv::Mat pan2 = cv::imread(argv[3]);

    if (ref.empty() || pan1.empty() || pan2.empty()) {
        std::cerr << "Error loading images.\n";
        return -1;
    }

    if (ref.size() != pan1.size()) {
        cv::resize(pan1, pan1, ref.size());
    }
    if (ref.size() != pan2.size()) {
        cv::resize(pan2, pan2, ref.size());
    }

    cv::Mat ref_gray, pan1_gray, pan2_gray;
    cv::cvtColor(ref, ref_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(pan1, pan1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(pan2, pan2_gray, cv::COLOR_BGR2GRAY);

    double ssim1 = computeSSIM(ref_gray, pan1_gray);
    double ssim2 = computeSSIM(ref_gray, pan2_gray);

    std::cout << "SSIM between reference and panorama_ORB: " << ssim1 << std::endl;
    std::cout << "SSIM between reference and panorama_SIFT: " << ssim2 << std::endl;

    return 0;
}
