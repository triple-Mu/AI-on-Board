//
// Created by ubuntu on 23-3-21.
//

#ifndef __DETDATA_H__
#define __DETDATA_H__

#include <algorithm>

const int BOX_COLORS[][3] = {
		{ 0, 114, 189 },
		{ 217, 83, 25 },
		{ 237, 177, 32 },
		{ 126, 47, 142 },
		{ 119, 172, 48 },
		{ 77, 190, 238 },
		{ 162, 20, 47 },
		{ 76, 76, 76 },
		{ 153, 153, 153 },
		{ 255, 0, 0 },
		{ 255, 128, 0 },
		{ 191, 191, 0 },
		{ 0, 255, 0 },
		{ 0, 0, 255 },
		{ 170, 0, 255 },
		{ 85, 85, 0 },
		{ 85, 170, 0 },
		{ 85, 255, 0 },
		{ 170, 85, 0 },
		{ 170, 170, 0 },
		{ 170, 255, 0 },
		{ 255, 85, 0 },
		{ 255, 170, 0 },
		{ 255, 255, 0 },
		{ 0, 85, 128 },
		{ 0, 170, 128 },
		{ 0, 255, 128 },
		{ 85, 0, 128 },
		{ 85, 85, 128 },
		{ 85, 170, 128 },
		{ 85, 255, 128 },
		{ 170, 0, 128 },
		{ 170, 85, 128 },
		{ 170, 170, 128 },
		{ 170, 255, 128 },
		{ 255, 0, 128 },
		{ 255, 85, 128 },
		{ 255, 170, 128 },
		{ 255, 255, 128 },
		{ 0, 85, 255 },
		{ 0, 170, 255 },
		{ 0, 255, 255 },
		{ 85, 0, 255 },
		{ 85, 85, 255 },
		{ 85, 170, 255 },
		{ 85, 255, 255 },
		{ 170, 0, 255 },
		{ 170, 85, 255 },
		{ 170, 170, 255 },
		{ 170, 255, 255 },
		{ 255, 0, 255 },
		{ 255, 85, 255 },
		{ 255, 170, 255 },
		{ 85, 0, 0 },
		{ 128, 0, 0 },
		{ 170, 0, 0 },
		{ 212, 0, 0 },
		{ 255, 0, 0 },
		{ 0, 43, 0 },
		{ 0, 85, 0 },
		{ 0, 128, 0 },
		{ 0, 170, 0 },
		{ 0, 212, 0 },
		{ 0, 255, 0 },
		{ 0, 0, 43 },
		{ 0, 0, 85 },
		{ 0, 0, 128 },
		{ 0, 0, 170 },
		{ 0, 0, 212 },
		{ 0, 0, 255 },
		{ 0, 0, 0 },
		{ 36, 36, 36 },
		{ 73, 73, 73 },
		{ 109, 109, 109 },
		{ 146, 146, 146 },
		{ 182, 182, 182 },
		{ 219, 219, 219 },
		{ 0, 114, 189 },
		{ 80, 183, 189 },
		{ 128, 128, 0 }
};

const char* NAMES[] = {
		"person", "bicycle", "car", "motorcycle", "airplane",
		"bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird",
		"cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack",
		"umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat",
		"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
		"wine glass", "cup", "fork", "knife", "spoon",
		"bowl", "banana", "apple", "sandwich", "orange",
		"broccoli", "carrot", "hot dog", "pizza", "donut",
		"cake", "chair", "couch", "potted plant", "bed",
		"dining table", "toilet", "tv", "laptop", "mouse",
		"remote", "keyboard", "cell phone", "microwave", "oven",
		"toaster", "sink", "refrigerator", "book", "clock",
		"vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

const int MASK_COLORS[][3] = {
		{ 255, 56, 56 },
		{ 255, 157, 151 },
		{ 255, 112, 31 },
		{ 255, 178, 29 },
		{ 207, 210, 49 },
		{ 72, 249, 10 },
		{ 146, 204, 23 },
		{ 61, 219, 134 },
		{ 26, 147, 52 },
		{ 0, 212, 187 },
		{ 44, 153, 168 },
		{ 0, 194, 255 },
		{ 52, 69, 147 },
		{ 100, 115, 255 },
		{ 0, 24, 236 },
		{ 132, 56, 255 },
		{ 82, 0, 133 },
		{ 203, 56, 255 },
		{ 255, 149, 200 },
		{ 255, 55, 199 }
};

const int KPT_COLORS[][3] = {
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 51, 153, 255 },
		{ 51, 153, 255 },
		{ 51, 153, 255 },
		{ 51, 153, 255 },
		{ 51, 153, 255 },
		{ 51, 153, 255 }
};

const int LIBM_COLORS[][3] = {
		{ 51, 153, 255 },
		{ 51, 153, 255 },
		{ 51, 153, 255 },
		{ 51, 153, 255 },
		{ 255, 51, 255 },
		{ 255, 51, 255 },
		{ 255, 51, 255 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 255, 128, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 0 }
};

const int SKELETON[][2] = {
		{ 16, 14 },
		{ 14, 12 },
		{ 17, 15 },
		{ 15, 13 },
		{ 12, 13 },
		{ 6, 12 },
		{ 7, 13 },
		{ 6, 7 },
		{ 6, 8 },
		{ 7, 9 },
		{ 8, 10 },
		{ 9, 11 },
		{ 2, 3 },
		{ 1, 2 },
		{ 1, 3 },
		{ 2, 4 },
		{ 3, 5 },
		{ 4, 6 },
		{ 5, 7 }
};

struct ResizeInfo
{
	float ratio_w;
	float ratio_h;
	int pad_l;
	int pad_t;
	int pad_r;
	int pad_b;

	ResizeInfo() = default;

	ResizeInfo(float rw, float rh, int pl, int pt, int pr, int pb) :
			ratio_w(rw), ratio_h(rh), pad_l(pl), pad_t(pt), pad_r(pr), pad_b(pb)
	{
	}

	ResizeInfo(float rw, float rh) :
			ratio_w(rw), ratio_h(rh), pad_l(0), pad_t(0), pad_r(0), pad_b(0)
	{
	}

	~ResizeInfo() = default;

};

struct Bbox
{
	float x{ 0.f };
	float y{ 0.f };
	float width{ 0.f };
	float height{ 0.f };

	Bbox() = default;

	Bbox(float xmin, float ymin, float w, float h)
			: x(xmin), y(ymin), width(w), height(h)
	{
	}

	float operator&(const Bbox& other) const
	{
		float x1 = std::max(this->x, other.x);
		float y1 = std::max(this->y, other.y);
		float x2 = std::min(this->x + this->width, other.x + other.width);
		float y2 = std::min(this->y + this->height, other.y + other.height);
		float intersection_area = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
		float union_area = this->width * this->height + other.width * other.height - intersection_area;
		if (union_area <= 0.f)
		{
			return 0.f;
		}
		return intersection_area / (union_area + 1e-12f);
	}

	float area() const
	{
		return this->width * this->height;
	}

	~Bbox() = default;
};

struct Object
{
	int label;
	float score;
	Bbox box;
};

#endif //__DETDATA_H__
