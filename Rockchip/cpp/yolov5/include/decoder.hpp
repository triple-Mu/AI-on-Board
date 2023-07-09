//
// Created by ubuntu on 23-5-31.
//

#ifndef __DECODER__
#define __DECODER__

#include <algorithm>
#include <cmath>
#include <vector>

#include "det_data.h"

template<typename T>
__inline__ static T clamp(T val, T min, T max)
{
    return val > min ? (val < max ? val : max) : min;
}

__inline__ static float fast_exp(float x)
{
    union {
        uint32_t i;
        float    f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

__inline__ static float un_sigmoid(float y)
{
    return -1.f * logf((1.f / y) - 1.f);
}

__inline__ static float fast_sigmoid(float x)
{
    return 1.f / (1.f + fast_exp(-x));
}

__inline__ static float fast_softmax(const float* src, float* dst, int length)
{
    const float alpha       = *std::max_element(src, src + length);
    float       denominator = 0;
    float       dis_sum     = 0;
    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
        dis_sum += (float)i * dst[i];
    }
    return dis_sum;
}

static void qsort_descent_inplace(std::vector<Object>& proposals, int left, int right)
{
    int   i = left;
    int   j = right;
    float p = proposals[(left + right) / 2].score;

    while (i <= j) {
        while (proposals[i].score > p) {
            i++;
        }

        while (proposals[j].score < p) {
            j--;
        }

        if (i <= j) {
            // swap
            std::swap(proposals[i], proposals[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(proposals, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(proposals, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void
batchednms(std::vector<Object>& proposals, std::vector<int>& indices, float iou_thres = 0.65f, bool agnostic = false)
{
    indices.clear();
    auto n = proposals.size();

    for (int i = 0; i < n; i++) {
        const Object& a = proposals[i];

        bool keep = true;

        for (auto& idx : indices) {
            const Object& b = proposals[idx];
            if (!agnostic && a.label != b.label) {
                continue;
            }
            if ((a.box & b.box) > iou_thres) {
                keep = false;
            }
        }
        if (keep) {
            indices.push_back(i);
        }
    }
}

static void generate_proposals_yolov5(const float*         feature,
                                      const float*         anchors,
                                      const int            feat_height,
                                      const int            feat_width,
                                      const int            stride,
                                      std::vector<Object>& proposals,
                                      float                conf_thres  = 0.25,
                                      int                  num_classes = 80)
{
    const int num_anchors         = 3;
    const int walk_through        = (5 + num_classes) * 3;
    float     usigmoid_conf_thres = un_sigmoid(conf_thres);
    for (int h = 0; h < feat_height; ++h) {
        for (int w = 0; w < feat_width; ++w) {
            const float* cur_ptr = feature + (h * feat_width + w) * walk_through;
            for (int na = 0; na < num_anchors; ++na) {
                const float  anchor_w       = anchors[na * 2];
                const float  anchor_h       = anchors[na * 2 + 1];
                const int    walk_anchor    = na * (5 + num_classes);
                const float* cur_anchor_ptr = cur_ptr + walk_anchor;
                if (cur_anchor_ptr[4] > usigmoid_conf_thres) {
                    const float* max_score_ptr = std::max_element(cur_anchor_ptr + 5, cur_anchor_ptr + 5 + num_classes);
                    float        score         = fast_sigmoid(cur_anchor_ptr[4]) * fast_sigmoid(*max_score_ptr);
                    if (score > conf_thres) {
                        float dx = fast_sigmoid(cur_anchor_ptr[0]);
                        float dy = fast_sigmoid(cur_anchor_ptr[1]);
                        float dw = fast_sigmoid(cur_anchor_ptr[2]);
                        float dh = fast_sigmoid(cur_anchor_ptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + (float)w) * (float)stride;
                        float pb_cy = (dy * 2.f - 0.5f + (float)h) * (float)stride;

                        float pb_w = std::pow(dw * 2.f, 2.f) * anchor_w * (float)stride;
                        float pb_h = std::pow(dh * 2.f, 2.f) * anchor_h * (float)stride;

                        float x1 = pb_cx - pb_w * 0.5f;
                        float y1 = pb_cy - pb_h * 0.5f;
                        float x2 = pb_cx + pb_w * 0.5f;
                        float y2 = pb_cy + pb_h * 0.5f;

                        int    label = (int)std::distance(cur_anchor_ptr + 5, max_score_ptr);
                        Object obj;
                        obj.label = label;
                        obj.score = score;
                        obj.box   = {x1, y1, x2 - x1, y2 - y1};
                        proposals.push_back(obj);
                    }
                }
            }
        }
    }
}
#endif  //__DECODER__
