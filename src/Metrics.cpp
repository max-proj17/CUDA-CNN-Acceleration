#include "Metrics.hpp"

// Helper function to calculate accuracy:

float calculateAccuracy(int true_positive, int true_negative, const torch::Tensor& targets) {
    return static_cast<float>(true_positive + true_negative) / targets.size(0);
}

// Helper function to calculate precision
float calculatePrecision(int true_positive, int false_positive) {
    return true_positive / static_cast<float>(true_positive + false_positive);
}

// Helper function to calculate recall
float calculateRecall(int true_positive, int false_negative) {
    return true_positive / static_cast<float>(true_positive + false_negative);
}

// Helper function to calculate F1 score
float calculateF1Score(float precision, float recall) {
    return 2 * (precision * recall) / (precision + recall);
}

// Function to calculate all metrics, update the Metrics struct and return it
Metrics calculateMetrics(const torch::Tensor& outputs, const torch::Tensor& targets) {
    Metrics metrics;

    auto prediction = outputs.argmax(1);
    auto true_positive = prediction.logical_and(targets).sum().item<int>();
    auto true_negative = prediction.logical_not().logical_and(targets.logical_not()).sum().item<int>();
    auto false_positive = prediction.logical_and(targets.logical_not()).sum().item<int>();
    auto false_negative = prediction.logical_not().logical_and(targets).sum().item<int>();

    metrics.accuracy = calculateAccuracy(true_positive, true_negative, targets);
    metrics.precision = calculatePrecision(true_positive, false_positive);
    metrics.recall = calculateRecall(true_positive, false_negative);
    metrics.f1_score = calculateF1Score(metrics.precision, metrics.recall);

    return metrics;
}
