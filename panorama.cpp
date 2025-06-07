#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using KeyPoints = std::vector<cv::KeyPoint>;
using Matches = std::vector<cv::DMatch>;

/**
 * @brief Фильтрация совпадений по правилу Лоу (Lowe's ratio test).
 * @param knnMatches Вектор списков совпадений, где каждый список содержит k
 * ближайших совпадений.
 * @param ratio Пороговое значение для фильтрации (по умолчанию 0.7).
 * @return Matches Отфильтрованные хорошие совпадения.
 */
Matches filterMatchesLowe(
    const std::vector<std::vector<cv::DMatch>> &knnMatches,
    float ratio = 0.5f) {
  Matches goodMatches;
  for (const auto &m : knnMatches) {
    if (m.size() >= 2 && m[0].distance < ratio * m[1].distance) {
      goodMatches.push_back(m[0]);
    }
  }
  return goodMatches;
}

/**
 * @brief Извлекает пары соответствующих точек из ключевых точек и совпадений.
 * @param matches Совпадения между двумя наборами ключевых точек.
 * @param kp1 Ключевые точки первого изображения.
 * @param kp2 Ключевые точки второго изображения.
 * @param pts1 Вектор для записи точек первого изображения (выходной параметр).
 * @param pts2 Вектор для записи точек второго изображения (выходной параметр).
 */
void extractMatchedPoints(const Matches &matches, const KeyPoints &kp1,
                          const KeyPoints &kp2, std::vector<cv::Point2f> &pts1,
                          std::vector<cv::Point2f> &pts2) {
  for (const auto &match : matches) {
    pts1.push_back(kp1[match.queryIdx].pt);
    pts2.push_back(kp2[match.trainIdx].pt);
  }
}

/**
 * @brief Построение панорамы из двух изображений с использованием гомографии.
 * @param img1 Первое изображение.
 * @param img2 Второе изображение.
 * @param H Матрица гомографии, отображающая img2 в систему координат img1.
 * @return Итоговое изображение-панорама.
 */
cv::Mat buildPanorama(const cv::Mat &img1, const cv::Mat &img2,
                      const cv::Mat &H) {
  std::vector<cv::Point2f> corners2 = {{0, 0},
                                       {(float)img2.cols, 0},
                                       {(float)img2.cols, (float)img2.rows},
                                       {0, (float)img2.rows}};
  std::vector<cv::Point2f> projectedCorners;
  cv::perspectiveTransform(corners2, projectedCorners, H);

  float minX = std::min(
      0.0f, std::min_element(projectedCorners.begin(), projectedCorners.end(),
                             [](auto &a, auto &b) { return a.x < b.x; })
                ->x);
  float minY = std::min(
      0.0f, std::min_element(projectedCorners.begin(), projectedCorners.end(),
                             [](auto &a, auto &b) { return a.y < b.y; })
                ->y);
  float maxX = std::max(
      (float)img1.cols,
      std::max_element(projectedCorners.begin(), projectedCorners.end(),
                       [](auto &a, auto &b) { return a.x < b.x; })
          ->x);
  float maxY = std::max(
      (float)img1.rows,
      std::max_element(projectedCorners.begin(), projectedCorners.end(),
                       [](auto &a, auto &b) { return a.y < b.y; })
          ->y);

  int width = static_cast<int>(std::ceil(maxX - minX));
  int height = static_cast<int>(std::ceil(maxY - minY));

  cv::Mat shift = cv::Mat::eye(3, 3, CV_64F);
  shift.at<double>(0, 2) = -minX;
  shift.at<double>(1, 2) = -minY;

  cv::Mat warped1, warped2;
  cv::warpPerspective(img1, warped1, shift, {width, height});
  cv::warpPerspective(img2, warped2, shift * H, {width, height});

  cv::Mat mask1 = cv::Mat::zeros(height, width, CV_8U);
  cv::Mat mask2 = cv::Mat::zeros(height, width, CV_8U);
  cv::warpPerspective(cv::Mat::ones(img1.size(), CV_8U), mask1, shift,
                      {width, height});
  cv::warpPerspective(cv::Mat::ones(img2.size(), CV_8U), mask2, shift * H,
                      {width, height});

  cv::Mat panorama(height, width, CV_8UC3, cv::Scalar::all(0));
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      bool in1 = mask1.at<uchar>(y, x);
      bool in2 = mask2.at<uchar>(y, x);
      if (in1 && in2) {
        cv::Vec3b c1 = warped1.at<cv::Vec3b>(y, x);
        cv::Vec3b c2 = warped2.at<cv::Vec3b>(y, x);
        cv::Vec3b c;
        c[0] = static_cast<uchar>((c1[0] + c2[0]) / 2);
        c[1] = static_cast<uchar>((c1[1] + c2[1]) / 2);
        c[2] = static_cast<uchar>((c1[2] + c2[2]) / 2);
        panorama.at<cv::Vec3b>(y, x) = c;
      } else if (in1) {
        panorama.at<cv::Vec3b>(y, x) = warped1.at<cv::Vec3b>(y, x);
      } else if (in2) {
        panorama.at<cv::Vec3b>(y, x) = warped2.at<cv::Vec3b>(y, x);
      }
    }
  }

  return panorama;
}

/**
 * @brief Сохраняет визуализации ключевых точек и совпадений для двух
 * изображений.
 * @param img1 Первое изображение.
 * @param img2 Второе изображение.
 * @param kp1 Ключевые точки первого изображения.
 * @param kp2 Ключевые точки второго изображения.
 * @param matches Совпадения между ключевыми точками.
 * @param detectorName Имя используемого детектора (для формирования имени
 * файлов).
 */
void saveVisualization(const cv::Mat &img1, const cv::Mat &img2,
                       const std::vector<cv::KeyPoint> &kp1,
                       const std::vector<cv::KeyPoint> &kp2,
                       const std::vector<cv::DMatch> &matches,
                       const std::string &detectorName) {
  cv::Mat img1_kp;
  cv::drawKeypoints(img1, kp1, img1_kp, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  std::string file1 = "Keypoints Image 1 (" + detectorName + ").png";
  cv::imwrite(file1, img1_kp);
  std::cout << "Saved keypoints visualization for Image 1 as: " << file1
            << std::endl;

  cv::Mat img2_kp;
  cv::drawKeypoints(img2, kp2, img2_kp, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  std::string file2 = "Keypoints Image 2 (" + detectorName + ").png";
  cv::imwrite(file2, img2_kp);
  std::cout << "Saved keypoints visualization for Image 2 as: " << file2
            << std::endl;

  cv::Mat matches_img;
  cv::drawMatches(img1, kp1, img2, kp2, matches, matches_img);
  std::string file3 = "Matches " + detectorName + ".png";
  cv::imwrite(file3, matches_img);
  std::cout << "Saved matches visualization as: " << file3 << std::endl;
}

/**
 * @brief Создает панораму из двух изображений с использованием заданного детектора признаков.
 * 
 * Выполняет следующие шаги:
 * - Детектирует ключевые точки и вычисляет дескрипторы на двух серых изображениях.
 * - Сопоставляет дескрипторы с помощью BFMatcher и фильтрует хорошие совпадения по критерию Лоу.
 * - Извлекает координаты совпавших ключевых точек.
 * - Строит гомографию методом RANSAC и, если она корректна, формирует панораму.
 * - Сохраняет полученную панораму в файл.
 * - Выводит визуализацию ключевых точек и совпадений.
 * 
 * @param detector Указатель на объект детектора и дескриптора признаков (например, SIFT или ORB).
 * @param gray1 Первое входное изображение в оттенках серого.
 * @param gray2 Второе входное изображение в оттенках серого.
 * @param img1 Первое входное цветное изображение.
 * @param img2 Второе входное цветное изображение.
 * @param methodName Имя используемого метода (например, "SIFT", "ORB") для сообщений и визуализации.
 * @param normType Тип нормы для BFMatcher (например, cv::NORM_L2 для SIFT, cv::NORM_HAMMING для ORB).
 * @param outputFilename Имя файла для сохранения результата панорамы.
 */
void createPanorama(const cv::Ptr<cv::Feature2D> &detector,
                    const cv::Mat &gray1, const cv::Mat &gray2,
                    const cv::Mat &img1, const cv::Mat &img2,
                    const std::string &methodName, int normType,
                    const std::string &outputFilename) {
  // Ключевые точки и дескрипторы
  KeyPoints kp1, kp2;
  cv::Mat des1, des2;
  detector->detectAndCompute(gray1, cv::noArray(), kp1, des1);
  detector->detectAndCompute(gray2, cv::noArray(), kp2, des2);

  // Сопоставление дескрипторов
  cv::BFMatcher matcher(normType);
  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher.knnMatch(des1, des2, knnMatches, 2);

  Matches matches = filterMatchesLowe(knnMatches);

  // Извлечение точек
  std::vector<cv::Point2f> pts1, pts2;
  extractMatchedPoints(matches, kp1, kp2, pts1, pts2);

  // Построение гомографии и панорамы
  if (pts1.size() >= 4) {
    std::vector<uchar> mask;
    cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, 5.0, mask);
    if (!H.empty()) {
      cv::Mat panorama = buildPanorama(img1, img2, H);
      cv::imwrite(outputFilename, panorama);
      std::cout << "Saved " << methodName << " panorama as: " << outputFilename
                << std::endl;
    }
  }
  else std::cerr << "Not enough matches to compute homography.\n";
  

  // Визуализация
  saveVisualization(img1, img2, kp1, kp2, matches, methodName);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2>\n";
    return -1;
  }

  cv::Mat img1 = cv::imread(argv[1]);
  cv::Mat img2 = cv::imread(argv[2]);
  if (img1.empty() || img2.empty()) {
    std::cerr << "Failed to load images.\n";
    return -1;
  }

  cv::Mat gray1, gray2;
  cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
  
  createPanorama(cv::ORB::create(), gray1, gray2, img1, img2, "ORB",
                 cv::NORM_HAMMING, "panorama_ORB.png");
  createPanorama(cv::SIFT::create(), gray1, gray2, img1, img2, "SIFT",
                 cv::NORM_L2, "panorama_SIFT.png");

  return 0;
}
