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

namespace lczero {


int const Q_TO_PROB_MODE = 1;
  // 1: e^(k * q)
  // 2: 1 / (1 + k (maxq - q))^2


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

inline float q_to_prob(const float q, const float max_q, const float q_concentration, int n, int parent_n) {
  return exp(q_concentration * (q - abs(max_q)/2)); // reduce the overflow risk. However, with the default q_concentration 36.2, overflow isn't possible since exp(36.2 * 1) is less than max float. TODO restrict the parameter so it can't overflow.
}

float computeChildWeights(NodeGlow* node, bool evaluation_weights, int node_n) {
  int n = 0;  
  float sum_of_P_of_expanded_nodes = 0.0;
  float maxq = -2.0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    float q = i->GetQ();
    if (q > maxq) maxq = q;
    n++;
  }
  if(evaluation_weights){
    // Just set w to normalised(q after q_to_prob)
    float sum_of_w_of_expanded_nodes = 0.0;  
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      i->SetW(q_to_prob(i->GetQ(), maxq, param_temperature - param_temperatureVisitOffset * node_n, i->GetN(), node_n));
      sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
      sum_of_w_of_expanded_nodes += i->GetW();
    }
    // Normalise to sum_of_P_of_expanded_nodes
    float normalising_factor = sum_of_P_of_expanded_nodes / sum_of_w_of_expanded_nodes;
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      i->SetW(i->GetW() * normalising_factor);
    }
    return(sum_of_P_of_expanded_nodes);
  } else {

    // Exploration weights

    // If there is only one expanded node, then let's just set w to policy and call it a day.
    // Unlike in UCB/MCTS the decision to expande a _new_ node isn't governed by this function.
    if(n == 1){
      NodeGlow *i = node->GetFirstChild();
      i->SetW(node->GetEdges()[i->GetIndex()].GetP());
      return(sum_of_P_of_expanded_nodes);
    }

    // The weight is simply the score used in UCT/MCTS but normalised to the sum of P of all expanded children.
    // Step 1.
    // Calculate the sum of P of all expanded children.
    // Calculate score for all expanded children
    // Also, keep track of the minimal score, which we need as offset later
    float sum_of_score_of_expanded_nodes = 0.0;
    // Unlike MCTS, we use W instead of Q. W is [0,1] so we can safely add them to a sum.
    // Right now CpuctFactor and CpuctBase is not accessible parameters, so define them here
    float CpuctFactor = 2;
    float CpuctBase = 19652;
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
      // Originally, I had i->GetQ() But that didn't work.
      // 3 for param_temperatureWinpctCutoff is just a number that works good here :-)
      i->SetW(param_temperatureWinpctCutoff * i->GetW() + node->GetEdges()[i->GetIndex()].GetP() * ( param_cpuct + CpuctFactor * log((node_n + CpuctBase)/CpuctBase) * sqrt(node_n) / ( 1 + i->GetN())));
      sum_of_score_of_expanded_nodes += i->GetW();
      n++;
    }
    // Step 2. Normalise the score
    float normalising_factor = sum_of_P_of_expanded_nodes / sum_of_score_of_expanded_nodes;
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      i->SetW(i->GetW() * normalising_factor);
    }
      return(sum_of_P_of_expanded_nodes);
    }
  }

float compute_q_and_weights(NodeGlow *node, int node_n) {
  // Evaluation weights      
  float total_children_weight = computeChildWeights(node, true, node_n);

  // Average Q START
  float q = (1.0 - total_children_weight) * node->GetOrigQ();
	for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    q -= i->GetW() * i->GetQ();
  }
  // Average Q STOP

  // Exploration weights
  total_children_weight = computeChildWeights(node, false, node_n);	

	return q;
}

}  // namespace lczero
