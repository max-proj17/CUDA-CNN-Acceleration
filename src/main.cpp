#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a tensor and print it.
    torch::Tensor tensor = torch::rand({ 2, 3 });
    std::cout << tensor << std::endl;

    // Check if CUDA is available and move the tensor to GPU
    if (torch::cuda::is_available()) {
        tensor = tensor.to(torch::kCUDA);
        std::cout << "CUDA is available! Tensor moved to GPU." << std::endl;
    }
    else {
        std::cout << "CUDA is not available." << std::endl;
    }
    // Wait for user input to close the console window
    std::cout << "Press ENTER to exit..." << std::endl;
    std::cin.get();
    return 0;
}
