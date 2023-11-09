#pragma once
#include <torch/torch.h>
#include "Train.hpp"
#include "Net.hpp"
#include "Metrics.hpp"
#include <chrono>
#include <vector>

template <typename DataLoader>
void train(Net& model,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    torch::Device device) {
    model.train();
    std::vector<torch::Tensor> epoch_predictions, epoch_targets;

    for (auto& batch : data_loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.squeeze();

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Reset gradients
        optimizer.zero_grad();
        // Execute the model on the input data
        auto output = model.forward(data);
        // Compute a loss value to judge the prediction of our model
        auto loss = torch::nll_loss(output, targets);
        // Compute gradients of the loss w.r.t. the parameters of our model
        loss.backward();
        // Update the parameters based on the calculated gradients
        optimizer.step();

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        // Collect predictions and targets to calculate metrics later
        epoch_predictions.push_back(output.detach());
        epoch_targets.push_back(targets.detach());

        // You might want to print out the loss here as well
        std::cout << "Loss: " << loss.item<float>() << ", Time per batch: " << diff.count() << " s" << std::endl;
    }

    // Combine predictions and targets from all batches
    auto all_predictions = torch::cat(epoch_predictions);
    auto all_targets = torch::cat(epoch_targets);

    // Calculate metrics for the epoch
    auto metrics = calculateMetrics(all_predictions, all_targets);
    std::cout << "Epoch Metrics: Accuracy: " << metrics.accuracy
        << ", Precision: " << metrics.precision
        << ", Recall: " << metrics.recall
        << ", F1 Score: " << metrics.f1_score << std::endl;
}
