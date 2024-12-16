#pragma once

#include <vector>

struct StateActionReward{
    std::vector<double> state; //inputs to network
    std::vector<double> action; //outputs of network
    double reward; //output of evaluatorFunction(state);

    StateActionReward(const std::vector<double>& state, const std::vector<double>& action, double reward);
};

using Trajectory = std::vector<StateActionReward>;

class PPO{
    public:
        PPO();
        static double loss();
        static double getDiscountedRewards(Trajectory trajectory, double discountFactor);
        static double BaselineEstimate();
    private:

};