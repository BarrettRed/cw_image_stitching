#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

// === Генераторы случайных чисел ===
std::mt19937 rng(std::random_device{}());

float randFloat(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

int randInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

// === Добавление гауссовского шума ===
cv::Mat addNoise(const cv::Mat& image, double amount, double& out_std) {
    cv::Mat noisy, noise(image.size(), CV_32FC3);
    cv::randn(noise, 0, amount);
    image.convertTo(noisy, CV_32FC3, 1.0 / 255.0);
    noisy = noisy + noise;
    cv::Mat result;
    cv::threshold(noisy, noisy, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(noisy, noisy, 0.0, 0.0, cv::THRESH_TOZERO);
    noisy.convertTo(result, CV_8UC3, 255.0);
    out_std = amount;
    return result;
}

// === Случайное изменение яркости и контраста ===
cv::Mat applyBrightnessContrast(const cv::Mat& image, float& out_alpha, int& out_beta) {
    out_alpha = randFloat(0.9f, 1.1f);
    out_beta = randInt(-20, 20);
    cv::Mat result;
    image.convertTo(result, -1, out_alpha, out_beta);
    return result;
}

// === Аффинное преобразование ===
cv::Mat applyRandomAffine(const cv::Mat& image, float& angle, float& scale, float& tx, float& ty) {
    int rows = image.rows, cols = image.cols;
    angle = randFloat(-5.0f, 5.0f);
    scale = randFloat(0.95f, 1.05f);
    tx = randFloat(-10.0f, 10.0f);
    ty = randFloat(-10.0f, 10.0f);

    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cols / 2, rows / 2), angle, scale);
    M.at<double>(0, 2) += tx;
    M.at<double>(1, 2) += ty;

    cv::Mat dst;
    cv::warpAffine(image, dst, M, image.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return dst;
}

// === Перспективное искажение ===
cv::Mat applyRandomPerspective(const cv::Mat& image, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst_pts) {
    int w = image.cols, h = image.rows;
    int shift = 20;

    auto rand_pt = [&](int x, int y) {
        return cv::Point2f(x + randInt(-shift, shift), y + randInt(-shift, shift));
    };

    src = { {0, 0}, {(float)w, 0}, {(float)w, (float)h}, {0, (float)h} };
    dst_pts = { rand_pt(0, 0), rand_pt(w, 0), rand_pt(w, h), rand_pt(0, h) };

    cv::Mat M = cv::getPerspectiveTransform(src, dst_pts);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, image.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return warped;
}

// === Основная функция подготовки изображений ===
void prepareSplitImages(const std::string& imagePath, float overlap_ratio, const std::string& outDir) {
    if (overlap_ratio <= 0.0f || overlap_ratio >= 1.0f) {
        throw std::invalid_argument("overlap_ratio должен быть между 0 и 1 (не включительно)");
    }

    if (!fs::exists(imagePath)) {
        throw std::runtime_error("Файл не найден: " + imagePath);
    }

    fs::create_directories(outDir);
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        throw std::runtime_error("Ошибка чтения изображения: " + imagePath);
    }

    int h = image.rows;
    int w = image.cols;
    int center = w / 2;
    int overlap = static_cast<int>(w * overlap_ratio);

    int left_start = 0;
    int left_end = center + overlap / 2;
    int right_start = center - overlap / 2;
    int right_end = w;

    cv::Mat left = image(cv::Rect(left_start, 0, left_end - left_start, h)).clone();
    cv::Mat right = image(cv::Rect(right_start, 0, right_end - right_start, h)).clone();

    // Преобразование левого изображения
    double noise_std_l;
    float alpha_l, angle_l, scale_l, tx_l, ty_l;
    int beta_l;
    left = addNoise(left, 0.02, noise_std_l);
    left = applyBrightnessContrast(left, alpha_l, beta_l);
    left = applyRandomAffine(left, angle_l, scale_l, tx_l, ty_l);

    // Преобразование правого изображения
    double noise_std_r;
    float alpha_r;
    int beta_r;
    std::vector<cv::Point2f> src_pts, dst_pts;
    right = addNoise(right, 0.02, noise_std_r);
    right = applyBrightnessContrast(right, alpha_r, beta_r);
    right = applyRandomPerspective(right, src_pts, dst_pts);

    // Сохранение изображений
    cv::imwrite(outDir + "/original.jpg", image);
    cv::imwrite(outDir + "/img1_left.jpg", left);
    cv::imwrite(outDir + "/img1_right.jpg", right);

    // Вывод параметров
    std::cout << "\nИзменения изображений:\n";
    std::cout << "--- LEFT ---\n";
    std::cout << "noise_std: " << noise_std_l << "\n";
    std::cout << "contrast_alpha: " << alpha_l << ", brightness_beta: " << beta_l << "\n";
    std::cout << "affine_angle_deg: " << angle_l << ", scale: " << scale_l
              << ", shift_x: " << tx_l << ", shift_y: " << ty_l << "\n";

    std::cout << "\n--- RIGHT ---\n";
    std::cout << "noise_std: " << noise_std_r << "\n";
    std::cout << "contrast_alpha: " << alpha_r << ", brightness_beta: " << beta_r << "\n";
    std::cout << "perspective_src: ";
    for (const auto& pt : src_pts) std::cout << "[" << pt.x << ", " << pt.y << "] ";
    std::cout << "\nperspective_dst: ";
    for (const auto& pt : dst_pts) std::cout << "[" << pt.x << ", " << pt.y << "] ";
    std::cout << std::endl;
}

int main() {
    try {
        prepareSplitImages("img1.jpg", 0.05f, "output");
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
