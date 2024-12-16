#include "ppo.h"

StateActionReward::StateActionReward(const std::vector<double>& state, const std::vector<double>& action, double reward) : state(state), action(action), reward(reward) {}

double PPO::getDiscountedRewards(Trajectory trajectory, double discountFactor){
    double total = 0;
    for (const auto& sar : trajectory){
        total += sar.reward;
    }
    return total;
}