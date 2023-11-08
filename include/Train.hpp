#pragma once
#include <torch/torch.h>
#include "Metrics.hpp"

void train(torch::nn::Module& model,
    torch::data::DataLoader<torch::data::datasets::MNIST>& data_loader,
    torch::optim::Optimizer& optimizer);

