#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/dataloader.h>
#include "Net.hpp"
#include "Train.hpp"
#include "Metrics.hpp"
#include <iostream>
#include <limits>

int main() {
    std::cout << "Select the device for training:" << std::endl;
    std::cout << "1. CPU" << std::endl;
    std::cout << "2. GPU" << std::endl;
    std::cout << "Enter your choice (1 or 2): ";
    int choice;
    std::cin >> choice;

    // Define the device
    torch::Device device(torch::kCPU); // Default to CPU
    if (choice == 2 && torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "CUDA is available! Training will proceed on GPU." << std::endl;
    }
    else {
        std::cout << "Training will proceed on CPU." << std::endl;
    }

    // Hyperparameters
    const int64_t batch_size = 64;

    auto train_dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // The return type of make_data_loader is a unique_ptr to a DataLoader
    auto data_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

    Net model;
    model.to(device);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // Call the train function template with the correct DataLoader type
    train<decltype(*data_loader)>(
        model, *data_loader, optimizer, device);

    std::cout << "Training complete. Press ENTER to exit..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return 0;
}
