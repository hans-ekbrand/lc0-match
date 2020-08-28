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
#include <valarray>

#include <gsl/gsl_cdf.h> // for the cummulative beta distribution gsl_cdf_beta_Q()
#include <gsl/gsl_errno.h> // for gsl_set_error_handler_off();
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h> // gsl_ran_beta()

// #include <functional> // for placeholders used with std::transform

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

  // 1. find out the sum of P for all extended nodes (return this)
  // 2. Obtain the probability of each node have the highest <true value>/<generating rate>
  // 2a. Set up a matrix with m rows and n columns, where n is the number of extended nodes and m is the number of samples you can afford.
  // 2b. Fill in the matrix column by column with samples
  // 2c. count number of wins row by row.
  // 2d. calculate the proportion of wins for each edge.
  // 3. Set each node's weight to that probability (normalised to the sum of P for all extended nodes, if used with glow, not normalised if used with a mcmc tree traverser).

  // Todo: re-use one random number generator, don't initiate a new one for every node!
  // If it's problematic to share one instance between threads, then at least use only one per thread.
  gsl_rng * r = gsl_rng_alloc(gsl_rng_taus);
  // LOGFILE << "the random number generation is in place; r points to it";
  int n = node->GetN();
  // LOGFILE << "number of children " << n ;
  std::valarray<double> matrix( n_samples * n );

  // LOGFILE << "entering bad loop";
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    // Set alpha and beta so that
    // alpha + beta = number of children - 2
    // alpha / (alpha + beta) == q
    float winrate = (i->GetQ() + 1) * 0.5;
    int visits = i->GetN() + 2; // count the pseudo visits too.
    // LOGFILE << "At child " << i->GetIndex() << " winrate=" << winrate << " visits=" << visits;
    double alpha = winrate * (visits - 2);
    double beta = visits - 2 - alpha;
    int j = 0;
    for(j = 0; j < n_samples; j++) {
      double foo = gsl_ran_beta(r, alpha, beta);    
      // LOGFILE << "generating samples using alpha=" << alpha << ", beta=" << beta << " result: " << foo << " like a boss";
      // j is the row number, i->GetIndex() is the colum number.
      matrix[j * i->GetIndex() + i->GetIndex()] = foo;
      // LOGFILE << "filling the matrix at position (" << j << ", " << i->GetIndex() << ") with value " << matrix[j * i->GetIndex() + i->GetIndex()] ;
    }
  }
  // Derive weights, by counting the number of wins for each edge (column).
  std::vector<float> win_counts(n);
  int my_winner;
  int j = 0;
  for(j = 0; j < n_samples; j++) {
    // Set up a slice: j=start, n_samples=size, n=stride
    std::valarray<double> my_row_val = matrix[std::slice(j,n,n_samples)];
    std::vector<float> my_row_vector(begin(my_row_val), end(my_row_val));
    my_winner = std::distance(my_row_vector.begin(), std::max_element(my_row_vector.begin(), my_row_vector.end()));
    // LOGFILE << "The winner in row " << j << " is edge: " << my_winner;
    win_counts[my_winner]++;
  }
  // rescale win_counts to win_rates (weights)
  float my_scaler = 1 / n_samples;
  std::vector<float> my_weights(n);
  j = 0; // reset a counter
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {  
    my_weights[j] = win_counts[j] * my_scaler;
    i->SetW(my_weights[j]);
    j++;
    // LOGFILE << "Child " << j << " has wins " << win_counts[j] << " and weight " << my_weights[j];
  }

  float sum_of_P_of_expanded_nodes = 0.0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
    n++;
  }
  
  if(normalise_to_sum_of_p){
    // float sum_of_w_of_expanded_nodes = 0.0;
    // float sum_of_weighted_p_and_q = 0.0;
    // to be written
  }

  return(sum_of_P_of_expanded_nodes); // with this, parent immediatly get q of first child.

  // float normalise_to_sum_of_p_scaler = sum_of_P_of_expanded_nodes / sum_of_w_of_expanded_nodes; // Avoid division in the loop, multiplication should be faster.
  // std::vector<float> weighted_p_and_q(n);
  // float relative_weight_of_p = 0;
  // float relative_weight_of_q = 0;
  // float CpuctBase = 7000;
  // int ii = 0;
  // for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), ii++) {
  //   i->SetW(i->GetW() * normalise_to_sum_of_p_scaler); // Normalise sum of Q to sum of P
  //   double cpuct_as_prob = 0;
  //   if(i->GetN() > (uint32_t)param_maxCollisionVisitsId){
  // 	double cpuct = log((node->GetN() + CpuctBase)/CpuctBase) * sqrt(log(node->GetN())/(1+i->GetN()));
  // 	// transform cpuct with the sigmoid function (the logistic function, 1/(1 + exp(-x))
  // 	cpuct_as_prob = 2 * 2 * param_cpuct * (1/(1 + exp(-cpuct)) - 0.5); // f(0) would be 0.5, we want it f(0) to be zero.
  //   }
  //   relative_weight_of_p = pow(i->GetN(), param_fpuValue_false) / (0.05 + i->GetN()) + cpuct_as_prob; // 0.05 is here to make Q have some influence after 1 visit.
  //   if (relative_weight_of_p > 1){
  //     relative_weight_of_p = 1;
  //     }
  //   relative_weight_of_q = 1 - relative_weight_of_p;
  //   // get an new term which should encourage exploration by multiplying both policy and q with this number.
  //   // or, for just add it in, the exploration bonus is for _everyone_.
  //   // old version
  //   // weighted_p_and_q[i] = relative_weight_of_q * node->GetEdges()[i].GetChild()->GetW() + relative_weight_of_p * node->GetEdges()[i].GetP() + node->GetEdges()[i].GetP() * search_->params_.GetCpuct() * log((node->GetN() + search_->params_.GetCpuctBase())/search_->params_.GetCpuctBase()) * sqrt(log(node->GetN())/(1+node->GetEdges()[i].GetChild()->GetN()));
  //   // new version
  //   weighted_p_and_q[ii] = relative_weight_of_q * i->GetW() + relative_weight_of_p * pow(node->GetEdges()[i->GetIndex()].GetP(), 0.85);
  //   sum_of_weighted_p_and_q += weighted_p_and_q[ii];
  // }

  // // Normalise the weighted sum Q + P to sum of P
  // float normalise_weighted_sum = sum_of_P_of_expanded_nodes / sum_of_weighted_p_and_q;
  // ii = 0;
  // for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), ii++) {
  //   i->SetW(weighted_p_and_q[ii] * normalise_weighted_sum);
  // }

}

float compute_q_and_weights(NodeGlow *node) {
  int number_of_samples = 100; // use this many samples to derive weights
  // LOGFILE << "in compute_q_and_weights";
  float total_children_weight = computeChildWeights(node, number_of_samples, false);

  // Average Q START
  float q = (1.0 - total_children_weight) * node->GetOrigQ();
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    q -= i->GetW() * i->GetQ();
  }
  // Average Q STOP
  return q;
}

}  // namespace lczero


