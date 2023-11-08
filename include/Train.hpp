#pragma once
#include <torch/torch.h>

// Declare the train function with the required arguments
void train(torch::nn::Module& model,
    torch::data::DataLoader<torch::data::datasets::MNIST>& data_loader,
    torch::optim::Optimizer& optimizer);
