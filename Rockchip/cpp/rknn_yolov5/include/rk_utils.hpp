//
// Created by ubuntu on 23-6-7.
//

#ifndef __RK_UTILS__
#define __RK_UTILS__

#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>

#include "checker.h"
#include "rknn_api.h"

typedef unsigned char uchar;
using timestamp = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>;

static void print_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index,
           attr->name,
           attr->n_dims,
           attr->dims[0],
           attr->dims[1],
           attr->dims[2],
           attr->dims[3],
           attr->n_elems,
           attr->size,
           get_format_string(attr->fmt),
           get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type),
           attr->zp,
           attr->scale);
}

static uchar* load_rknn_model(const std::string& rknn_path, int* size)
{
    std::ifstream file(rknn_path, std::ios::binary);
    CHECK(file.good());
    file.seekg(0, std::ios::end);
    *size = (int)file.tellg();
    file.seekg(0, std::ios::beg);
    uchar* model_buffer = new uchar[*size];
    CHECK(model_buffer != nullptr);
    file.read(reinterpret_cast<char*>(model_buffer), *size);
    file.close();
    return model_buffer;
}

class MovingAverageFPS {
public:
    explicit MovingAverageFPS(const int& size = 30)
    {
        this->size = size;
        this->sum  = 0.f;
    }

    float next(const float& val)
    {
        if (this->fps_queue.size() == size) {
            this->sum -= this->fps_queue.front();
            this->fps_queue.pop();
        }
        this->fps_queue.emplace(val);
        this->sum += val;
        return this->sum / this->fps_queue.size();
    }

    ~MovingAverageFPS() = default;

private:
    int               size;
    float             sum;
    std::queue<float> fps_queue;
};

static timestamp get_timestamp()
{
    return std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
}

static long get_count(timestamp start, timestamp end)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

#endif  //__RK_UTILS__
