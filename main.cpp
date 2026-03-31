#include <iostream>
#include <vector>
#include <chrono>

#define epsilon 1e-6
#define max_iterations 10000

class RawMath {

    public: 
        static double exp(double x) {
            double res = 1.0;
            double term = 1.0;
            for (int i = 1; i <= 15; i++) {
                term *= x / i;
                res += term;
            }
            return res;
        }

        static double abs(double x) {
            return x >= 0 ? x : -x;
        }

        static double sigmoid(double z) {
            return 1.0 / (1.0 + exp(-z));
        }

        static double square(double x) {
            return x * x;
        }

};


class Engine {
    private: 
        double lr;
        double lambda = 0;

        public:
        Engine(double learning_rate) : lr(learning_rate) {}
        

        double predict(double x, double weight) {
            return RawMath::sigmoid(x * weight);
        }

        double g(double target, double pred) {    
            return RawMath::square(target * pred) - 1;
        }   

        void train(double x, double & weight, double target) {

            double pred = predict(x, weight);
            double velocity = 0;
            int iterations = 0;
            
            while (RawMath::abs(target - pred) > 0.01 && max_iterations > iterations ) {
                iterations++;

                // LOSS = 1/2 (u)^2, where u = (pred - target)                
                double error = pred - target;

                // Langrandgian function L(x, lambda) = h(x) * lambda
                lambda = lambda + lr * g(target, pred);

                // chain rule for sigmoid derivative
                double h_derivative =  (2 * (target - pred) * (pred * (1 - pred)) * x);
                double gradient = error * (pred * (1.0 - pred)) * x + lambda * h_derivative;

                double second_var = ((pred * (1.0 - pred)) * x) * ((pred * (1.0 - pred)) * x);

                velocity = velocity + (lr * (gradient / (second_var + epsilon)));
                
                weight = weight - velocity;
                
                pred = predict(x, weight);

                pred = std::max(0.00001, std::min(0.99999, pred));
                
                std::cout << "Pred: " << pred  << " | Target: " << target 
                  << " | New Weight: " << weight << '\n';
    
            }
    }    
};

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> weights = {0.1, 0.1, 0.1};
    std::vector<double> targets = {1.0, 0.7, 0.5};

    Engine engine(0.1);

    for (int i = 0; i < weights.size(); ++i) {
        engine.train(1.0, weights[i], targets[i]);
    }

    for (int i = 0; i < weights.size(); i++) 
        std::cout << "Target : " << targets[i] << " " << "Weight : " << weights[i] << " " << '\n';

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
     
    std::cout << "Training took: " << duration.count() << " microseconds" << "\n";
}
