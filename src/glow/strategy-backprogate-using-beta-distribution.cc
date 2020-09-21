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
#include <math.h>

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




// calculates approximate probability for each of a set beta distributed random variables to have the highest value
// constant parameter: MAX_INTERVALS is the maximum number of intervals used for the computation
// input:
// qnw.size() is the number of random variables
// qnw[i].first is the q for variable i, i.e. 2 * mu - 1
// qnw[i].second is the n for variable i, i.e. nu
// beta distributions have prior (1,1)
// qnw.size() >= 0
// for each q, -1 <= q <= 1
// foe each n, 0 <= n or n = +infinity
// output:
// qnw[i].first is relative probability of variable i to be highest
//  0 <= qnw[i].first <= 1
// qnw[i].second is unchanged
// return value is sum of probabilities
//  0 < sum <= 1 when qnw.size() > 0, typically 0.95-0.98
//  sum = 0 when qnw.size() = 0
// time complexity: O(qnw.size() * min(qnw.size(), MAX_INTERVALS))
// error: mean error per variable probability is typically 0.02 for 5 variables, 0.01 for 10 variables
//  (provided MAX_INTERVALS >= 10)

#define MAX_INTERVALS 20

int compare_doubles(const void* a, const void* b)
{
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;
 
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

struct sortunit {
 double v;
 int idx;
};

double calc_beta_distr_pwin(std::vector<std::pair<double, double>> &qnw) {

 const int nvar = qnw.size();

 struct sortunit maxs[nvar];
 double logmaxs[nvar][2];
 double albes[nvar][2];

 for (int j = 0; j < nvar; j++) {
  const double q = ldexp(qnw[j].first + 1.0, -1);
  maxs[j].v = q;
  maxs[j].idx = j;

  logmaxs[j][0] = log2(q);
  logmaxs[j][1] = log2(1.0 - q);
  // const double size = qnw[j].second;
  const double size = param_maxCollisionVisitsId * qnw[j].second;
  albes[j][0] = q * size;
  albes[j][1] = size - albes[j][0];
 }

 qsort(maxs, nvar, sizeof(struct sortunit), compare_doubles);

 const int nintval = MAX_INTERVALS < nvar ? MAX_INTERVALS : nvar;

 double p[nvar][nintval];

 const int maxsoff = nvar - nintval;

 for (int j = 0; j < nvar; j++) {
  double sum = 0.0;
  double llim = 0.0;
  const int idxj = maxs[j].idx;
  if (isinf(qnw[idxj].second)) {
   for (int k = 0; k < nintval; k++) {
    const double rlim = (k == nintval - 1) ? 1.0 : (0.5 * (maxs[maxsoff + k].v + maxs[maxsoff + k + 1].v));
    const int idxk = maxs[maxsoff + k].idx;
    if (k == 0 && j <= maxsoff || idxk == idxj) {
     p[j][k] = rlim - llim;
    } else {
     p[j][k] = qnw[idxj].first == qnw[idxk].first ? (rlim - llim) : 0.0;
    }
    sum += p[j][k];
    llim = rlim;
   }
  } else {
   for (int k = 0; k < nintval; k++) {
    const double rlim = (k == nintval - 1) ? 1.0 : (0.5 * (maxs[maxsoff + k].v + maxs[maxsoff + k + 1].v));
    const int idxk = maxs[maxsoff + k].idx;
    if (k == 0 && j <= maxsoff || idxk == idxj) {
     p[j][k] = rlim - llim;
    } else {
     p[j][k] =
      exp2(
       (albes[idxj][0] == 0.0 ? 0.0 : ((logmaxs[idxk][0] - logmaxs[idxj][0]) * albes[idxj][0])) +
       (albes[idxj][1] == 0.0 ? 0.0 : ((logmaxs[idxk][1] - logmaxs[idxj][1]) * albes[idxj][1]))
      ) * (rlim - llim);
    }
    sum += p[j][k];
    llim = rlim;
   }
  }
  for (int k = 0; k < nintval; k++) {
   p[j][k] /= sum;
  }
 }

 for (int j = 0; j < nvar; j++) {
  albes[j][0] = 0.0;
  albes[j][1] = 0.0;
 }
 
 for (int k = 0; k < nintval; k++) {
  double factor = 1.0;
  for (int j = 0; j < nvar; j++) {
   albes[j][0] += 0.5 * p[j][k];
   factor *= albes[j][0];
  }
  for (int j = 0; j < nvar; j++) {
   if (albes[j][0] > 0.0) albes[j][1] += p[j][k] * factor / albes[j][0];
   albes[j][0] += 0.5 * p[j][k];
  }
 }

 double sum = 0.0;
 for (int j = 0; j < nvar; j++) {
  qnw[maxs[j].idx].first = albes[j][1];
  sum += albes[j][1];
 }
 return sum;
}


double computeChildWeights(NodeGlow* node) {

  int n = node->GetNumChildren();

  std::vector<std::pair<double, double>> qnw(n);
  
  double sum_of_P_of_expanded_nodes = 0.0;
  int j = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), j++) {
    qnw[j].first = (double)i->GetQ();
    qnw[j].second = i->IsTerminal() ? INFINITY : (double)i->GetN();
    sum_of_P_of_expanded_nodes += node->GetEdges()[i->GetIndex()].GetP();
  }
  double sumw = calc_beta_distr_pwin(qnw);
  double normf = sum_of_P_of_expanded_nodes / sumw;
  j = 0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling(), j++) {
    i->SetW((float)(normf * qnw[j].first));
  }

  return sum_of_P_of_expanded_nodes;
}

float compute_q_and_weights(NodeGlow *node) {
  double total_children_weight = computeChildWeights(node);
  if((total_children_weight >= 1.00014) | (total_children_weight < 0)){
    LOGFILE << "total weight is weird " << total_children_weight;
  }
  assert((total_children_weight <= 1.00014) & (total_children_weight >= 0));
  // weighted average Q START
  assert((node->GetOrigQ() <= 1) & (node->GetOrigQ() >= -1));
  double q = (1.0 - total_children_weight) * (double)node->GetOrigQ();
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    assert((i->GetW() <= 1) & (i->GetW() >= 0));
    assert((i->GetQ() <= 1) & (i->GetQ() >= -1));
    q -= (double)i->GetW() * (double)i->GetQ();
  }
  // weighted average Q STOP
  assert((q <= 1) & (q >=-1));
  return (float)q;
}

}  // namespace lczero
