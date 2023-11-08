#include <torch/torch.h>
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

    // MNIST dataset
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Normalize<>(0.1307, 0.3081)).map(
                torch::data::transforms::Stack<>()),
        batch_size);

    // Define your model and optimizer here
    Net model;
    model.to(device);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));

    // Call the train function
    train(model, *data_loader, optimizer);

    std::cout << "Training complete. Press ENTER to exit..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return 0;
}
