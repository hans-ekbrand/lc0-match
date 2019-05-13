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


//////////////////////////////////////////////////////////////////////////////
// Distribution
//////////////////////////////////////////////////////////////////////////////

	// Parameters we use
	// FpuValue: policy_weight_exponent 0.59
  	// Temperature: q_concentration 36.2
        // MaxCollisionVisits: number of sub nodes before exploration encouragement kicks in. 900 is reasonable, but for long time control games perhaps better leave it at 1.
  	// TemperatureVisitOffset: coefficient by which q_concentration is reduced per subnode. 0.0000082
	// TemperatureWinpctCutoff: factor to boost cpuct * policy = explore moves with high policy but low q. reasonable value: 3-5 (when TemperatureVisitOffset is non-zero lower, say 0.5)
	// Cpuct: factor to boost cpuct = explore moves regardless of their policy reasonable value: 0.003

  // The implementation in this file is taken from rev 88439502739789a375cf532558bd69b9caece998 "Back to cpuct again, this is known well playing baseline." which I think is the best playing version at least for up to 100.000 nodes.
  // This version do not separate evaluation from exploration, and ignores MaxCollisionVisits
  // With OOE we haven't clopped values, only tested out at few (all with 4 threads):
  // Sensible values on CPuct with this version is 0.2 (+5-3=22 at 6.000 nodes, 4 threads) For 3.000 nodes use CPuct = 0 (+7-3=20). @12.000 CPuct = 0.15 only even +3-3=24. @36.000 Cpuct = 0.1 even: +12-12=76. @72.000 worse +6-17=77.

  // Another version which should be tested, and might be better than this is 5add726b69131a2cf0ffd6b8ba6c08e5097cd4db

inline float q_to_prob(const float q, const float max_q, const float q_concentration, int n, int parent_n) {
  switch (Q_TO_PROB_MODE) {
  case 1: {
    // double my_q_concentration_ = 35.2;
    // double my_q_concentration_ = 40;
    // When a parent is well explored, we need to reward exploration instead of (more) exploitation.
    // A0 and Lc0 does this by increase policy on children with relatively few visits.
    // Let's try to do that instead by decreasing q_concentration as parent N increases.
    // At higher visits counts, our policy term has no influence anymore.
    // I've modelled a function after the dynamic cpuct invented by DeepMind, so that our function decreases by half at the same number parent nodes as the the dynamic cpuct is doubled (for zero visit at the child). We reward exploration regardless of number of child visits, which might not be as effective as their strategy, but let's give it a go.
    // return exp(q_concentration * (0.246 + (1 - 0.246) / pow((1 + parent_n / 30000), 0.795)) * (q - abs(max_q)/2)); // reduce the overflow risk.
    // if(parent_n > 100000){
    //   // Reduce q_concentration to 35.3 by 1E6 and 34.8 by 3E6.
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n)/2.5 - 4.6);
    //   // Reduce q_concentration to 34.7 by 1E6 and 33.9 by 3E6.      
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n)/1.5 - 7.67);
    //   // Reduce q_concentration to 34.3 by 1E6 and 33.4 by 3E6.            
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n)/1.2 - 9.59);            
    //   // Reduce q_concentration to 33.9 by 1E6 and 32.8 by 3E6.            
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n) - 11.51);
    //   // cyclic, mean = q_concentration - amplitude
    //   float amplitude = 1.1;
    //   float dynamic_q_concentration = (1 + cos(3.141592 * n / 50000)) * amplitude + q_concentration - 2 * amplitude;
    //   return exp(dynamic_q_concentration * (q - abs(max_q)/2)); // reduce the overflow risk.
    // } else {
      return exp(q_concentration * (q - abs(max_q)/2)); // reduce the overflow risk. However, with the default q_concentration 36.2, overflow isn't possible since exp(36.2 * 1) is less than max float. TODO restrict the parameter so that it cannot overflow and remove this division.
    // }
  };
  case 2: {
    float x = 1.0 + 20.0 * (max_q - q);
    return pow(x, -2.0);
  };
  };
}


float computeChildWeights(NodeGlow* node, bool evaluation_weights, int node_n) {
  int n = 0;
  float maxq = -2.0;
	for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    float q = i->GetQ();
    if (q > maxq) maxq = q;
		n++;
  }

  float sum_of_P_of_expanded_nodes = 0.0;
  float sum_of_w_of_expanded_nodes = 0.0;
  float sum_of_weighted_p_and_q = 0.0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    float w = q_to_prob(i->GetQ(), maxq, param_temperature, i->GetN(), node_n);
    i->SetW(w);
    sum_of_w_of_expanded_nodes += w;
    sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
  }
  float normalise_to_sum_of_p = sum_of_P_of_expanded_nodes / sum_of_w_of_expanded_nodes; // Avoid division in the loop, multiplication should be faster.
  std::vector<float> weighted_p_and_q(n);
  float relative_weight_of_p = 0;
  float relative_weight_of_q = 0;
  int ii = 0;
  float heighest_weight = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), ii++) {
    i->SetW(i->GetW() * normalise_to_sum_of_p); // Normalise sum of Q to sum of P

    relative_weight_of_p = pow(i->GetN(), param_fpuValue_false) / (0.05 + i->GetN()); // 0.05 is here to make Q have some influence after 1 visit.
    relative_weight_of_q = 1 - relative_weight_of_p;

    weighted_p_and_q[ii] = relative_weight_of_q * i->GetW() + relative_weight_of_p * node->GetEdges()[i->GetIndex()].GetP();
    sum_of_weighted_p_and_q += weighted_p_and_q[ii];
    if(weighted_p_and_q[ii] > heighest_weight){
      heighest_weight = weighted_p_and_q[ii];
    }
  }

  // Normalise the weighted sum Q + P to sum of P
  float normalise_weighted_sum = sum_of_P_of_expanded_nodes / sum_of_weighted_p_and_q;
  if(!evaluation_weights){
    // It is up to the caller to decide when exploration weights are wanted.
    // mimic a/b pruning by encouraging exploration of unexpanded nodes for beta at low depth. Ideally only in our PV, but to some extent that should happen automatically, right?
    // Do the children represent moves by beta? If not, punish all children by normalising to a value lower than sum of P.
    // Is this node deep into the tree? If so punish all children by normalising to a value lower than sum of P.
    // Children represent moves by beta if depth is 1?
    int depth = 0;
    NodeGlow *node_tmp = node;
    while (node_tmp != nullptr) {
      node_tmp = node_tmp->GetParent();
      depth++;
    }
    if(depth % 2 != 0){
      // The children are moves by Alpha
      // LOGFILE << "Alpha: Depth: " << depth << " Move: " << node->GetEdges()[node->GetFirstChild()->GetIndex()].GetMove(false).as_string() << " Reducing exploration weight for a alpha move by multiplying with " << normalise_weighted_sum * pow(param_temperatureWinpctCutoff, depth) << " instead of: " << normalise_weighted_sum;
      // Discourage exploring alphas moves
      if(param_temperatureWinpctCutoff != 1.0){
	normalise_weighted_sum = normalise_weighted_sum * pow(param_temperatureWinpctCutoff, log2(depth));
      }
    }
    // Penalize nodes deep in the tree:
    // LOGFILE << "Depth: " << depth << " Move: " << node->GetEdges()[node->GetFirstChild()->GetIndex()].GetMove(false).as_string() << " Reducing exploration weight according to depth node by multiplying with  " << normalise_weighted_sum * pow(param_cpuct, depth) << " instead of: " << normalise_weighted_sum << " depth: " << depth;
    if(param_cpuct != 1.0){
      normalise_weighted_sum = normalise_weighted_sum * pow(param_cpuct, log2(depth));
    }
  }
  ii = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), ii++) {
    i->SetW(weighted_p_and_q[ii] * normalise_weighted_sum);
  }

  return(sum_of_P_of_expanded_nodes);
}

float compute_q_and_weights(NodeGlow *node, int node_n) {
  float total_children_weight = computeChildWeights(node, true, node_n);

//  if (total_children_weight < 0.0 || total_children_weight - 1.0 > 1.00012) {
//    std::cerr << "total_children_weight: " << total_children_weight << "\n";
//    abort();
//  }
//  float totw = 0.0;
//  for (int i = 0; i < node->GetNumChildren(); i++) {
//    float w = node->GetEdges()[i].GetChild()->GetW();
//    if (w < 0.0) {
//      std::cerr << "w: " << w << "\n";
//      abort();
//    }
//    totw += w;
//  }
//  if (abs(total_children_weight - totw) > 1.00012) {
//    std::cerr << "total_children_weight: " << total_children_weight << ", totw: " << total_children_weight << "\n";
//    abort();
//  }

  // Average Q START
  float q = (1.0 - total_children_weight) * node->GetOrigQ();
	for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    q -= i->GetW() * i->GetQ();
  }
  // Average Q STOP

	if(node_n > 10){
	  total_children_weight = computeChildWeights(node, false, node_n);
	}

	return q;
}

}  // namespace lczero
