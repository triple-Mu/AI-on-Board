#include "rk_utils.hpp"
#include "rknet_cv.hpp"

int main(int argc, char** argv)
{
    std::string model_name{argv[1]};
    std::string video_name{argv[2]};
    std::string window_name = "on detection";

    cv::VideoCapture cap(video_name);
    cv::namedWindow(window_name, cv::WINDOW_FREERATIO);
    cv::resizeWindow(window_name, 1280, 960);
    cv::moveWindow(window_name, 300, 300);
    auto detector = new RKYOLOv5(model_name, 0);
    auto mfps     = MovingAverageFPS(30);

    cv::Mat             image;
    std::vector<Object> objects;
    while (cap.isOpened()) {
        cap >> image;
        timestamp start = get_timestamp();
        detector->forward(image, objects, true);
        timestamp end  = get_timestamp();
        auto      cost = 1e6f / get_count(start, end);
        draw_fps(image, mfps.next(cost));
        cv::imshow(window_name, image);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    delete detector;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
