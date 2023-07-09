//
// Created by ubuntu on 23-5-31.
//

#ifndef __RKRUNNER__
#define __RKRUNNER__

#include "checker.h"
#include "det_data.h"
#include "im2d.h"
#include "rga.h"
#include "rknn_api.h"

#include "decoder.hpp"
#include "opencv2/opencv.hpp"
#include "rk_utils.hpp"

#define NUM_CLASSES 80
#define CONF_THRES 0.25
#define IOU_THRES 0.65

static void draw_fps(cv::Mat& image, float fps)
{
    char text[32];
    sprintf(text, "FPS=%.2f", fps);

    int      baseLine   = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = image.cols - label_size.width;

    cv::rectangle(image, {x, y, label_size.width, label_size.height + baseLine}, {255, 255, 255}, -1);

    cv::putText(image, text, {x, y + label_size.height}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0});
}

static void draw_on_image(cv::Mat& image, const std::vector<Object>& objects)
{
    int baseLine = 0;
    for (auto& object : objects) {
        const int& cate_idx = object.label;
        cv::Scalar bbox_color(BOX_COLORS[cate_idx][0], BOX_COLORS[cate_idx][1], BOX_COLORS[cate_idx][2]);

        const char* name = NAMES[cate_idx];
        cv::Rect    bbox(object.box.x, object.box.y, object.box.width, object.box.height);
        cv::rectangle(image, bbox, bbox_color, 2);
        char text[256];
        sprintf(text, "%s %.1f%%", name, object.score * 100);

        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        int      x          = bbox.x;
        int      y          = bbox.y + 1;

        if (y > image.rows) {
            y = image.rows;
        }

        cv::rectangle(image, {x, y, label_size.width, label_size.height + baseLine}, {0, 0, 255}, -1);

        cv::putText(image, text, {x, y + label_size.height}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}

class RKYOLOv5 {
public:
    RKYOLOv5(const std::string& rknn_path, int core_id);

    ~RKYOLOv5();

    void preprocess(const cv::Mat& image);

    int forward(cv::Mat& image, std::vector<Object>& objects, bool draw);

public:
    int   mNet_h;
    int   mNet_w;
    int   mNumInputs;
    int   mNumOutputs;
    float mAnchors[3][6] = {{1.25, 1.625, 2.0, 3.75, 4.125, 2.875},
                            {1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375},
                            {3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875}};

    // using for multithread
    cv::Mat             mImage;
    std::vector<Object> mObjects;

private:
    rknn_tensor_attr mInput_attr;
    rknn_tensor_attr mOutput_attrs[3];

    rknn_input  mInput;
    rknn_output mOutputs[3];

    rga_buffer_t src;
    rga_buffer_t dst;

    im_rect src_rect;
    im_rect dst_rect;

    rknn_context mContext;
    void*        resize_buf = nullptr;
};

RKYOLOv5::RKYOLOv5(const std::string& rknn_path, int core_id)
{
    int    model_size;
    uchar* model_buffer = load_rknn_model(rknn_path, &model_size);
    CHECK(rknn_init(&this->mContext, model_buffer, model_size, RKNN_FLAG_COLLECT_PERF_MASK, nullptr) >= 0);
    free(model_buffer);
    rknn_core_mask core_mask;
    if (core_id == -1) {
        core_mask = RKNN_NPU_CORE_AUTO;
    }
    else if (core_id % 3 == 0) {
        core_mask = RKNN_NPU_CORE_0;
    }
    else if (core_id % 3 == 1) {
        core_mask = RKNN_NPU_CORE_1;
    }
    else {
        core_mask = RKNN_NPU_CORE_2;
    }
    CHECK(rknn_set_core_mask(this->mContext, core_mask) >= 0);

    rknn_input_output_num io_num;
    CHECK(rknn_query(this->mContext, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) >= 0);
    this->mNumInputs  = io_num.n_input;
    this->mNumOutputs = io_num.n_output;

    memset(&this->mInput_attr, 0x00, sizeof(rknn_tensor_attr));
    memset(&this->mInput, 0x00, sizeof(rknn_input));

    this->mInput_attr.index = 0;
    CHECK(rknn_query(this->mContext, RKNN_QUERY_INPUT_ATTR, &this->mInput_attr, sizeof(rknn_tensor_attr)) >= 0);

    memset(&this->mOutput_attrs[0], 0x00, sizeof(rknn_tensor_attr));
    memset(&this->mOutput_attrs[1], 0x00, sizeof(rknn_tensor_attr));
    memset(&this->mOutput_attrs[2], 0x00, sizeof(rknn_tensor_attr));

    memset(&this->mOutputs[0], 0x00, sizeof(rknn_output));
    memset(&this->mOutputs[1], 0x00, sizeof(rknn_output));
    memset(&this->mOutputs[2], 0x00, sizeof(rknn_output));

    this->mOutput_attrs[0].index = 0;
    this->mOutput_attrs[1].index = 1;
    this->mOutput_attrs[2].index = 2;

    CHECK(rknn_query(this->mContext, RKNN_QUERY_OUTPUT_ATTR, &this->mOutput_attrs[0], sizeof(rknn_tensor_attr)) >= 0);
    CHECK(rknn_query(this->mContext, RKNN_QUERY_OUTPUT_ATTR, &this->mOutput_attrs[1], sizeof(rknn_tensor_attr)) >= 0);
    CHECK(rknn_query(this->mContext, RKNN_QUERY_OUTPUT_ATTR, &this->mOutput_attrs[2], sizeof(rknn_tensor_attr)) >= 0);

    if (this->mInput_attr.fmt == RKNN_TENSOR_NCHW) {
        this->mNet_h = this->mInput_attr.dims[2];
        this->mNet_w = this->mInput_attr.dims[3];
    }
    else {
        this->mNet_h = this->mInput_attr.dims[1];
        this->mNet_w = this->mInput_attr.dims[2];
    }

    this->mInput.index = 0;
    this->mInput.type  = RKNN_TENSOR_UINT8;
    this->mInput.size  = 1 * 3 * this->mNet_h * this->mNet_w;
    this->mInput.fmt          = RKNN_TENSOR_NHWC;
    this->mInput.pass_through = 0;

    memset(&this->src, 0x00, sizeof(this->src));
    memset(&this->dst, 0x00, sizeof(this->dst));
    memset(&this->src_rect, 0x00, sizeof(this->src_rect));
    memset(&this->dst_rect, 0x00, sizeof(this->dst_rect));
}

RKYOLOv5::~RKYOLOv5()
{
    CHECK(rknn_destroy(this->mContext) >= 0);
    if (this->resize_buf != nullptr) {
        free(this->resize_buf);
        this->resize_buf = nullptr;
    }
}

void RKYOLOv5::preprocess(const cv::Mat& image)
{
    int img_width  = image.cols;
    int img_height = image.rows;

    cv::Mat input_image;
    cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);

    if (img_width != this->mNet_w || img_height != this->mNet_h) {

        if (this->resize_buf == nullptr) {
            this->resize_buf = malloc(this->mNet_h * this->mNet_w * 3 * sizeof(uint8_t));
        }
        memset(this->resize_buf, 0x00, this->mNet_h * this->mNet_w * 3 * sizeof(uint8_t));

        this->src = wrapbuffer_virtualaddr(input_image.data, img_width, img_height, RgaSURF_FORMAT::RK_FORMAT_RGB_888);
        this->dst =
            wrapbuffer_virtualaddr(this->resize_buf, this->mNet_w, this->mNet_h, RgaSURF_FORMAT::RK_FORMAT_RGB_888);
        CHECK(imcheck(src, dst, src_rect, dst_rect) == IM_STATUS::IM_STATUS_NOERROR);
        CHECK(imresize(src, dst) == IM_STATUS::IM_STATUS_SUCCESS);
        this->mInput.buf = this->resize_buf;
    }
    else {
        this->mInput.buf = input_image.data;
    }
}

int RKYOLOv5::forward(cv::Mat& image, std::vector<Object>& objects, bool draw)
{
    if (image.empty()) {
        return 1;
    }
    int img_width  = image.cols;
    int img_height = image.rows;
    this->preprocess(image);
    ResizeInfo resize_info = ResizeInfo(this->mNet_w / (float)img_width, this->mNet_h / (float)img_height);


    CHECK(rknn_inputs_set(this->mContext, this->mNumInputs, &this->mInput) >= 0);


    this->mOutputs[0].want_float = 1;
    this->mOutputs[1].want_float = 1;
    this->mOutputs[2].want_float = 1;

    CHECK(rknn_run(this->mContext, nullptr) >= 0);
    CHECK(rknn_outputs_get(this->mContext, this->mNumOutputs, this->mOutputs, nullptr) >= 0);

    objects.clear();
    std::vector<Object> proposals;

    int stride = 4;
    for (int i = 0; i < this->mNumOutputs; i++) {
        float* feature = (float*)this->mOutputs[i].buf;
        stride <<= 1;
        generate_proposals_yolov5(feature,
                                  this->mAnchors[i],
                                  this->mNet_h / stride,
                                  this->mNet_w / stride,
                                  stride,
                                  proposals,
                                  CONF_THRES,
                                  NUM_CLASSES);
    }

    std::vector<int> indices;

    batchednms(proposals, indices, IOU_THRES, false);

    for (auto& idx : indices) {
        const Object& pro   = proposals[idx];
        auto&         bbox  = pro.box;
        float         x1    = bbox.x;
        float         y1    = bbox.y;
        float         x2    = bbox.x + bbox.width;
        float         y2    = bbox.y + bbox.height;
        float         score = pro.score;
        int           label = pro.label;

        x1 = (x1 - resize_info.pad_l) / resize_info.ratio_w;
        y1 = (y1 - resize_info.pad_t) / resize_info.ratio_h;
        x2 = (x2 - resize_info.pad_l) / resize_info.ratio_w;
        y2 = (y2 - resize_info.pad_t) / resize_info.ratio_h;

        x1 = clamp(x1, 1.f, img_width - 1.f);
        y1 = clamp(y1, 1.f, img_height - 1.f);
        x2 = clamp(x2, 1.f, img_width - 1.f);
        y2 = clamp(y2, 1.f, img_height - 1.f);

        Object object;
        object.score = score;
        object.label = label;
        object.box   = {x1, y1, x2 - x1, y2 - y1};
        objects.push_back(object);
    }
    if (draw) {
        draw_on_image(image, objects);
    }
    return 0;
}

#endif  //__RKRUNNER__
