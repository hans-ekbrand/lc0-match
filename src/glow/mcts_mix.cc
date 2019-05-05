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

  // This version is even with MCTS @144.000 nodes on 4 cards: +11-10=79 with the following settings: cpuct=0.01, maxcollisionvisits=3000

	// Parameters we use
	// FpuValue: policy_weight_exponent 0.59
  	// Temperature: q_concentration 36.2
        // MaxCollisionVisits: number of sub nodes before exploration encouragement kicks in. 900 is reasonable, but for long time control games perhaps better leave it at 1.
	// Cpuct: factor to boost cpuct = explore moves regardless of their policy reasonable value: 0.01

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


float computeChildWeights(NodeGlow* node, bool evaluation_weights) {
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
    float w = q_to_prob(i->GetQ(), maxq, param_temperature, i->GetN(), node->GetN());
    i->SetW(w);
    sum_of_w_of_expanded_nodes += w;
    sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
  }
  float normalise_to_sum_of_p = sum_of_P_of_expanded_nodes / sum_of_w_of_expanded_nodes; // Avoid division in the loop, multiplication should be faster.
  std::vector<float> weighted_p_and_q(n);
  float relative_weight_of_p = 0;
  float relative_weight_of_q = 0;
  float CpuctBase = 7000;
  int ii = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), ii++) {
    i->SetW(i->GetW() * normalise_to_sum_of_p); // Normalise sum of Q to sum of P
    double cpuct_as_prob = 0;
    if(i->GetN() > (uint32_t)param_maxCollisionVisitsId){
	double cpuct = log((node->GetN() + CpuctBase)/CpuctBase) * sqrt(log(node->GetN())/(1+i->GetN()));
	// transform cpuct with the sigmoid function (the logistic function, 1/(1 + exp(-x))
	if(!evaluation_weights){
	  cpuct_as_prob = 2 * 2 * param_cpuct * (1/(1 + exp(-cpuct)) - 0.5); // f(0) would be 0.5, we want it f(0) to be zero.	  
	}
    }
    relative_weight_of_p = pow(i->GetN(), param_fpuValue_false) / (0.05 + i->GetN()) + cpuct_as_prob; // 0.05 is here to make Q have some influence after 1 visit.
    if (relative_weight_of_p > 1){
      relative_weight_of_p = 1;
      }
    relative_weight_of_q = 1 - relative_weight_of_p;
    // get an new term which should encourage exploration by multiplying both policy and q with this number.
    // or, for just add it in, the exploration bonus is for _everyone_.
    // old version
    // weighted_p_and_q[i] = relative_weight_of_q * node->GetEdges()[i].GetChild()->GetW() + relative_weight_of_p * node->GetEdges()[i].GetP() + node->GetEdges()[i].GetP() * search_->params_.GetCpuct() * log((node->GetN() + search_->params_.GetCpuctBase())/search_->params_.GetCpuctBase()) * sqrt(log(node->GetN())/(1+node->GetEdges()[i].GetChild()->GetN()));
    // new version
    if(evaluation_weights){
      weighted_p_and_q[ii] = relative_weight_of_q * i->GetW() + relative_weight_of_p * node->GetEdges()[i->GetIndex()].GetP() + node->GetEdges()[i->GetIndex()].GetP() * 0.1 * log((node->GetN() + CpuctBase)/CpuctBase) * sqrt(log(node->GetN())/(1+i->GetN())); // copied from baseline
    } else {
      if(node->GetN() > 400000){ // When parent has 400.000 visits use MCTS score as weight
	float CpuctFactor = 2;
	float CpuctBase = 19652;
	float CPuct = 3;
	weighted_p_and_q[ii] = node->GetEdges()[i->GetIndex()].GetP() * ( CPuct + CpuctFactor * log((node->GetN() + CpuctBase)/CpuctBase) * sqrt(node->GetN()) / ( 1 + i->GetN()));
      } else {
	weighted_p_and_q[ii] = relative_weight_of_q * i->GetW() + relative_weight_of_p * pow(node->GetEdges()[i->GetIndex()].GetP(), 0.75);
      }
    }
    sum_of_weighted_p_and_q += weighted_p_and_q[ii];
  }

  // Normalise the weighted sum Q + P to sum of P
  float normalise_weighted_sum = sum_of_P_of_expanded_nodes / sum_of_weighted_p_and_q;
  ii = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), ii++) {
    i->SetW(weighted_p_and_q[ii] * normalise_weighted_sum);
  }

  return(sum_of_P_of_expanded_nodes);
}

float compute_q_and_weights(NodeGlow *node) {
  float total_children_weight = computeChildWeights(node, true);

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

  total_children_weight = computeChildWeights(node, false);	

	return q;
}

}  // namespace lczero
