/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 Hans Ekbrand, Fredrik Lindblad and The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "glow/strategy.h"

#include <iostream>
#include <valarray> // for easily finding winners from the beta sampling

// use kumaraswamy instead of beta, sample from uniform [0,1]
#include <iostream>
#include <random>

#include <cassert>  // assert() used for debugging during development

namespace lczero {

float param_temperature;
float param_fpuValue_false;
int param_maxCollisionVisitsId;
float param_temperatureVisitOffset;
float param_temperatureWinpctCutoff;
float param_cpuct;

void set_strategy_parameters(const SearchParams *params) {
	param_temperature = params->GetTemperature();
	param_fpuValue_false = params->GetFpuValue(false);
	param_maxCollisionVisitsId = params->GetMaxCollisionVisitsId();
	param_temperatureVisitOffset = params->GetTemperatureVisitOffset();
	param_temperatureWinpctCutoff = params->GetTemperatureWinpctCutoff();
	param_cpuct = params->GetCpuct();
}

  // What is the intuition behind the ChildWeight and what is the intended use case?
  // ChildWeight for an edge should reflect the probability that this edge represents the best move.

  // As such it serves the purpose of weight when Q is backpropagated
  // to the parent. In this usage, we want the precision of the weight
  // to as high as possible, since errors would multiplicate upwards,
  // ie. we optimise for exploitation and use a rather high n_samples,
  // (or even an exact analytical solution, if that is faster) and
  // ignore all other parameters.

  // For the purpose of tree-traversing, we need exploration and in particular we need mechanisms that depend on
  // 1. Policy
  // 2. Number of visits (After many visits, additional visits does not bring the same amount of information, so some decay function is reasonable here).

  // 1. Policy
  // High policy should be reflected in a high number of visits - until the absolute number of visits predicted by policy is reached. One way of advancing edges with high P is a two-stage sampling procedure:
  //   1. Use policy to sample the first finalist
  //   2. Use Q to sample the second finalist
  //   Draw a winner from the two finalist using the beta distribution of q (with a rather low n).

  // a decay of concentration of visits on the best path as the number
  // of visits to the best path grow, since the additional nodes will
  // have a diminishing affect on the move selection.

  // n_samples how many samples to draw from each extended edge using the beta distribution

  // The use Policy: Treat P as a priori distribution of visits and add the real visits to it as they come. When traversing the tree,
  // policy_predicts_k_nodes Should roughly reflect the average number nodes per move in training games, e.g. 800

  // int share_visits_after_m_visits, int x0, float spare_L_proportion_to_self, float steepness

  // Tree traversing is sufficiently different from backpropagating that we can have two different functions. computeChildWeight() is for backpropagating and could be analytical (or use sampling)

  float computeChildWeights(NodeGlow* node, int n_samples, bool normalise_to_sum_of_p) {

    // LOGFILE << "Entered computeChildweights";

  // 1. find out the sum of P for all extended nodes (return this)
  // 2. Obtain the probability of each node have the highest <true value>/<generating rate>
  // 2a. Set up a matrix with m rows and n columns, where n is the number of extended nodes and m is the number of samples you can afford.
  // 2b. Fill in the matrix column by column with samples
  // 2c. count number of wins row by row.
  // 2d. calculate the proportion of wins for each edge.
  // 3. Set each node's weight to that probability (normalised to the sum of P for all extended nodes, if used with glow, not normalised if used with a mcmc tree traverser).

  // Todo: make sure the use the random number generator is thread safe.

  int n = node->GetNumChildren();

  // If there is only one expanded child, this child has the weight of policy.
  if(n == 1){
    float policy = node->GetEdges()[0].GetP();
    node->GetFirstChild()->SetW(policy);
    return(policy);
  }

  // // Not sure why this stops early. is W set to 0 elsewhere?
  // // All edges are extended, then we know that all children have got a weight, and we may ignore setting/updating the weights to save computation time.
  // if(node->GetNextUnexpandedEdge() == node->GetNumEdges()){
  //   if(node->GetN() % 5 > 0){ // Only update if the parent has a number visits that is divisible with 5
  //     return(1);
  //   }
  // }

  std::valarray<double> matrix( n_samples * n );

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);  

  int column = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    // Set alpha and beta so that
    // alpha + beta = number of visits + 2
    // alpha / (alpha + beta) == q
    float winrate = (i->GetQ() + 1) * 0.5;
    // float winrate = -0.5 * i->GetQ();
    int realvisits = i->GetN() + 2; // count the pseudo visits too.
    int visits = 0;
    if (realvisits > 100){
      visits = 100;
    } else {
      visits = realvisits;
    }
    float alpha = winrate * visits;
    float beta = visits - alpha;
    int row = 0;
    for(row = 0; row < n_samples; row++) {
      double foo = pow(1 - pow((1 - distribution(generator)), (1/beta)), (1/alpha));
      matrix[column + row * n] = foo;
      // LOGFILE << "At edge " << i->GetIndex() << " row " << row << " column " << column << "(position " << column + row * n << ") generating samples using alpha=" << alpha << ", beta=" << beta << " result: " << foo << " like a boss";
    }
    column++;
  }

  // Derive weights, by counting the number of wins for each edge (column).
  std::vector<int> win_counts(n);
  int my_winner;
  int j = 0;
  for(j = 0; j < n_samples * n; j = j + n) {
    // Set up a slice: j=start, n_samples=size, n=stride
    std::valarray<double> my_row_val = matrix[std::slice(j,n,1)];
    std::vector<double> my_row_vector(begin(my_row_val), end(my_row_val));
    my_winner = std::distance(my_row_vector.begin(), std::max_element(my_row_vector.begin(), my_row_vector.end()));
    // if(n == 3){
    //   LOGFILE << "competition: " << my_row_vector[0] << " "<< my_row_vector[1] << " "<< my_row_vector[2] << " The winner is " << my_winner;
    // }
    win_counts[my_winner]++;
  }

  // rescale win_counts to win_rates (weights)
  float my_scaler = 1.0f / n_samples;
  std::vector<float> my_weights(n);
  j = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {  
    my_weights[j] = win_counts[j] * my_scaler;

    float policy_decay_factor = 3.5f;
    float alpha_prior = node->GetEdges()[i->GetIndex()].GetP() * policy_decay_factor;
    float beta_prior = policy_decay_factor - alpha_prior;

    int realvisits = i->GetN();
    int visits = 0;
    if (realvisits > 100){
      visits = 100;
    } else {
      visits = realvisits;
    }
    float alpha = my_weights[j] * visits;
    float beta = visits - alpha;
    float E = (alpha + alpha_prior) / (alpha + alpha_prior + beta + beta_prior);

    // Terminal nodes does not have policy, so don't try this on a terminal node.
    if(i->IsTerminal()){
      // // Terminal nodes will not get visits anyway, just set the weight to whatever rescaled Q is
      // i->SetW(i->GetOrigQ());
    } else {					
      // i->SetW(my_weights[j]);
      i->SetW(E);
    }
    // LOGFILE << "Child " << node->GetEdges()[i->GetIndex()].GetMove(false).as_string()  << " has policy " << node->GetEdges()[i->GetIndex()].GetP() << " and " << i->GetN() << " true visits and " << win_counts[j] << " out of " << n_samples << " trials and would have got weight " << my_weights[j] << " but with the policy prior the weight becomes: " << E;
    j++;
  }

  float sum_of_P_of_expanded_nodes = 0.0;
  float sum_of_weights = 0.0;  
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
    sum_of_weights += i->GetW();
    n++;
  }
  float my_final_scaler = sum_of_P_of_expanded_nodes / sum_of_weights;

  // This is only necessary when there are unexpanded edges
  // if(normalise_to_sum_of_p & (node->GetNumChildren() < node->GetNumEdges())){
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      // i->SetW(i->GetW() * sum_of_P_of_expanded_nodes);
      i->SetW(i->GetW() * my_final_scaler);
    }
  // }

  return(sum_of_P_of_expanded_nodes); // with this, parent immediatly get q of first child.

}

float compute_q_and_weights(NodeGlow *node) {
  int number_of_samples = 500; // use this many samples to derive weights
  float total_children_weight = computeChildWeights(node, number_of_samples, true);

  // Average Q START
  float q = (1.0 - total_children_weight) * node->GetOrigQ();
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    assert(i->GetW() <= 1 & i->GetW() >= 0);
    assert(i->GetQ() <= 1 & i->GetQ() >= 0);
    q -= i->GetW() * i->GetQ();
  }
  // Average Q STOP
  assert(q <= 1 & q >=0);
  return q;
}

}  // namespace lczero
