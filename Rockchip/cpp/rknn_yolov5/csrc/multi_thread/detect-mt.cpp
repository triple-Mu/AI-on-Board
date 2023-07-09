#include "ThreadPool.h"
#include "rknet_cv.hpp"

#define NUM_THREAD 8

int main(int argc, char** argv)
{
    std::string model_name{argv[1]};
    std::string video_name{argv[2]};
    std::string window_name = "on detection";

    size_t           counter = 0;
    cv::VideoCapture cap(video_name);
    cv::namedWindow(window_name, cv::WINDOW_FREERATIO);
    cv::resizeWindow(window_name, 1280, 960);
    cv::moveWindow(window_name, 300, 300);

    std::vector<RKYOLOv5*>       detectors;
    ThreadPool                   pool(NUM_THREAD);
    std::queue<std::future<int>> futures;

    // 初始化
    for (int i = 0; i < NUM_THREAD; i++) {
        RKYOLOv5* ptr = new RKYOLOv5(model_name, i);
        detectors.push_back(ptr);
        cap >> ptr->mImage;
        futures.push(
            pool.enqueue(&RKYOLOv5::forward, std::ref(*ptr), std::ref(ptr->mImage), std::ref(ptr->mObjects), true));
    }

    timestamp start = get_timestamp();
    timestamp end;

    while (cap.isOpened()) {
        if (futures.front().get() != 0) {
            break;
        }
        futures.pop();
        cv::imshow(window_name, detectors[counter % NUM_THREAD]->mImage);
        if (cv::waitKey(1) == 'q')  // 延时1毫秒,按q键退出
            break;
        if (!cap.read(detectors[counter % NUM_THREAD]->mImage))
            break;
        auto cnt = counter++ % NUM_THREAD;
        futures.push(pool.enqueue(&RKYOLOv5::forward,
                                  std::ref(*detectors[cnt]),
                                  std::ref(detectors[cnt]->mImage),
                                  std::ref(detectors[cnt]->mObjects),
                                  true));

        if (counter % 60 == 0) {
            end = get_timestamp();
            printf("FPS every 60 frames is: %4.2f\n", 6e7f / get_count(start, end));
            start = end;
        }
    }

    // 释放剩下的资源
    while (!futures.empty()) {
        if (futures.front().get())
            break;
        futures.pop();
    }
    for (int i = 0; i < NUM_THREAD; i++) {
        delete detectors[i];
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
