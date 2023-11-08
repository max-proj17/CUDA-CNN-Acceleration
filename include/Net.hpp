#pragma once
#include <torch/torch.h>

struct Net : torch::nn::Module {
    Net() {
        // Initialize CNN layers
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)));
        fc1 = register_module("fc1", torch::nn::Linear(320, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
        // Max pooling over a (2, 2) window
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
        x = x.view({ -1, 320 }); // Flatten the tensor
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }

    // CNN layers
    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};
