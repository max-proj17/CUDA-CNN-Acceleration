#pragma once
#include <torch/torch.h>

struct Metrics {
	float accuracy;
	float precision;
	float recall;
	float f1_score;
	// ... other metrics and functions to calculate them
};

Metrics calculateMetrics(const torch::Tensor& outputs, const torch::Tensor& targets);