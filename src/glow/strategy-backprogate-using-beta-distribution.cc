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
// #include <gsl/gsl_errno.h> // for gsl_set_error_handler_off();
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h> // gsl_ran_beta()

// #include <functional> // for placeholders used with std::transform

namespace lczero {

  gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);

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

  // Todo: make sure the use the random number generator is thread safe.

  int n = node->GetNumChildren();

  // If there is only one expanded child, this child has the weight of policy.
  if(n == 1){
    float policy = node->GetEdges()[0].GetP();
    node->GetFirstChild()->SetW(policy);
    return(policy);
  }

  // gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);

  // LOGFILE << "number of children " << n ;

  // std::valarray<int> my_matrix( 3 * 4);
  // for (int i=0; i < 3 * 4 ; ++i) my_matrix[i]=i;

// slice( std::size_t start, std::size_t size, std::size_t stride );
// size_t star : is the index of the first element in the selection
// size_t size : is the number of elements in the selection
// stride : is the span that separates the elements selected.

  // // Extract the second row
  // std::slice_array<int> test = my_matrix[std::slice(3,1,3)];

  // printf("testing matrix slices, this should be 3 but it is %d, the first column on the second row, first row 0, 1, 2, second row 3, 4, 5", test[0]);

  std::valarray<double> matrix( n_samples * n );

  // double my_matrix[200][30];

  // const gsl_rng_type * T;
  // gsl_rng * r;
  // gsl_rng_env_setup(); // read environmental variables to set the type 
  // T = gsl_rng_default;
  // r = gsl_rng_alloc (T);

  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    // Set alpha and beta so that
    // alpha + beta = number of visits + 2
    // alpha / (alpha + beta) == q
    float winrate = (i->GetQ() + 1) * 0.5;
    // float winrate = -0.5 * i->GetQ();
    int visits = i->GetN() + 2; // count the pseudo visits too.
    // LOGFILE << "At child " << i->GetIndex() << " winrate=" << winrate << " visits=" << visits;
    double alpha = winrate * visits;
    double beta = visits - alpha;
    int row = 0;
    for(row = 0; row < n_samples; row++) {
      // LOGFILE << "generating samples using alpha=" << alpha << ", beta=" << beta;
      double foo = gsl_ran_beta(r, alpha, beta);
      // LOGFILE << "At edge " << i->GetIndex() << " row " << row << "(position " << i->GetIndex() + row * n << ") generating samples using alpha=" << alpha << ", beta=" << beta << " result: " << foo << " like a boss";
      // i->GetIndex() is the colum number.
      // matrix[row * i->GetIndex() + i->GetIndex()] = foo;
      matrix[i->GetIndex() + row * n] = foo;
      // LOGFILE << "filling matrix at position (" << i->GetIndex() + row * n << ") with value " << matrix[i->GetIndex() + row * n] ;
      // my_matrix[row][i->GetIndex()] = foo;
      // LOGFILE << "filling my_matrix at position [" << row << "][" << i->GetIndex() << "] with value " << my_matrix[row][i->GetIndex()];
    }
  }

  // gsl_rng_free (r);

  // LOGFILE << matrix[2] << " should be second result for edge 0";
  // LOGFILE << my_matrix[1][0] << " should again be second result edge 0";

  // Derive weights, by counting the number of wins for each edge (column).
  std::vector<int> win_counts(n);
  std::vector<int> real_win_counts(n);
  int my_winner;
  // int real_my_winner;
  int j = 0;
  // int k = 0;
  for(j = 0; j < n_samples * n; j = j + n) {
    // Set up a slice: j=start, n_samples=size, n=stride
    std::valarray<double> my_row_val = matrix[std::slice(j,n,1)];
    std::vector<float> my_row_vector(begin(my_row_val), end(my_row_val));
    my_winner = std::distance(my_row_vector.begin(), std::max_element(my_row_vector.begin(), my_row_vector.end()));
    // LOGFILE << "competition: " << my_row_vector[0] << " against " << my_row_vector[1] << " winner: " << my_winner;
    win_counts[my_winner]++;
  }
  // for(j = 0; j < n_samples; j++) {
  //   std::vector<float> my_real_row_vector(n); // store the samples here
  //   for(k=0; k < n; k++){
  //     my_real_row_vector[k] = my_matrix[j][k];
  //   }
  //   // identify the highes sample in each row (j)
  //   real_my_winner = std::distance(my_real_row_vector.begin(), std::max_element(my_real_row_vector.begin(), my_real_row_vector.end()));
  //   real_win_counts[real_my_winner]++;
  //   // LOGFILE << "The winner in row " << j << " is edge: " << my_winner;
  // }

  // rescale win_counts to win_rates (weights)
  float my_scaler = 1.0f / n_samples;
  // LOGFILE << "n_samples: " << n_samples << " my_scaler:" << my_scaler;
  std::vector<float> my_weights(n);
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {  
    my_weights[i->GetIndex()] = win_counts[i->GetIndex()] * my_scaler;
    i->SetW(my_weights[i->GetIndex()]);
    // LOGFILE << "Child " << node->GetEdges()[i->GetIndex()].GetMove(false).as_string()  << " has index " << i->GetIndex() << " and " << i->GetN() << " true visits and " << win_counts[i->GetIndex()] << " wins and " << real_win_counts[i->GetIndex()] << " real wins and weight " << my_weights[i->GetIndex()];
  }

  float sum_of_P_of_expanded_nodes = 0.0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
    n++;
  }

  // This is only necessary when there are unexpanded edges
  if(normalise_to_sum_of_p & (node->GetNumChildren() < node->GetNumEdges())){
    my_scaler = sum_of_P_of_expanded_nodes;
    // LOGFILE << "Normalising weights to sum to " << sum_of_P_of_expanded_nodes;
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      // LOGFILE << "scaling the weight for child" << i->GetIndex() << " from " << i->GetW() << " to " << i->GetW() * my_scaler;
      i->SetW(i->GetW() * my_scaler);
    }
  }

  return(sum_of_P_of_expanded_nodes); // with this, parent immediatly get q of first child.

}

float compute_q_and_weights(NodeGlow *node) {
  int number_of_samples = 30; // use this many samples to derive weights
  float total_children_weight = computeChildWeights(node, number_of_samples, true);

  // Average Q START
  float q = (1.0 - total_children_weight) * node->GetOrigQ();
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    q -= i->GetW() * i->GetQ();
  }
  // Average Q STOP
  return q;
}

}  // namespace lczero


