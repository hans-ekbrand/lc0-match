/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "mcts_replace/search.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <atomic> // global depth is an atomic int
#include <numeric> // accumulate()

#include "neural/encoder.h"

namespace lczero {

namespace {

// Alternatives:

bool const MULTIPLE_NEW_SIBLINGS = false;
  // If true then multiple children of the same node can be created and evaluated in the same batch.
  //  When using this alternative the Qs of the existing children plus the Ps of all unexpanded edges must sum up to 1. This must be fullfilled by the computeChildWeights function.
  // If true then at most the first (the one with highest network computed P) unexpanded edge is expanded in the same batch.
  //  When using this alternative the weight of expanding next edge 1 - sum of Qs of the existing children.
bool const INITIAL_MAX_P_IS_1 = true;
  // If true the the max weight of a node node is set to 1
  // If false the the max weight of a node node is set to the leftmost/highest P value
int const DISTRIBUTION_FUNCTION = 1;
  // 0 - Hans' distribution. Only implemented for non MULTIPLE_NEW_SIBLINGS alternative.
  // 1 - Plain non-adaptive exponential distribution. Uses cpuct parameter as concentration parameter, so higher values give higher trees and lower give wider trees.

const int kUciInfoMinimumFrequencyMs = 500;

}  // namespace


std::string SearchLimits_revamp::DebugString() const {
  std::ostringstream ss;
  ss << "visits:" << visits << " playouts:" << playouts << " depth:" << depth
     << " infinite:" << infinite;
  if (search_deadline) {
    ss << " search_deadline:"
       << FormatTime(SteadyClockToSystemClock(*search_deadline));
  }
  return ss.str();
}


Search_revamp::Search_revamp(const NodeTree_revamp& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits_revamp& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    :
      root_node_(tree.GetCurrentHead()),
      //~ syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      limits_(limits),
      params_(options),
      start_time_(std::chrono::steady_clock::now()),
      initial_visits_(root_node_->GetN()),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback)
    {}

int64_t Search_revamp::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}


void Search_revamp::StartThreads(size_t how_many) {
  how_many = 1;  // single threaded so far
//  LOGFILE << "Letting " << how_many << " threads create " << limits_.visits << " nodes each\n";

  Mutex::Lock lock(threads_mutex_);
  for (int i = 0; i < (int)how_many; i++) {
    threads_.emplace_back([this]()
      {
        SearchWorker_revamp worker(this, root_node_, params_);
        worker.RunBlocking();
      }
    );
  }
}

//~ bool Search_revamp::IsSearchActive() const {
  //~ Mutex::Lock lock(counters_mutex_);
  //~ return !stop_;
//~ }

void Search_revamp::WatchdogThread() {
  //~ SearchWorker_revamp worker(this, root_node_);
  //~ worker.RunBlocking();

  //~ while (IsSearchActive()) {
    //~ {
      //~ using namespace std::chrono_literals;
      //~ constexpr auto kMaxWaitTime = 100ms;
      //~ constexpr auto kMinWaitTime = 1ms;
      //~ Mutex::Lock lock(counters_mutex_);
      //~ auto remaining_time = limits_.time_ms >= 0
                                //~ ? (limits_.time_ms - GetTimeSinceStart()) * 1ms
                                //~ : kMaxWaitTime;
      //~ if (remaining_time > kMaxWaitTime) remaining_time = kMaxWaitTime;
      //~ if (remaining_time < kMinWaitTime) remaining_time = kMinWaitTime;
      //~ // There is no real need to have max wait time, and sometimes it's fine
      //~ // to wait without timeout at all (e.g. in `go nodes` mode), but we
      //~ // still limit wait time for exotic cases like when pc goes to sleep
      //~ // mode during thinking.
      //~ // Minimum wait time is there to prevent busy wait and other thread
      //~ // starvation.
      //~ watchdog_cv_.wait_for(lock.get_raw(), remaining_time,
                            //~ [this]()
                                //~ NO_THREAD_SAFETY_ANALYSIS { return stop_; });
    //~ }
    //~ MaybeTriggerStop();
  //~ }
  //~ MaybeTriggerStop();
}


/*
void Search_revamp::RunBlocking(size_t threads) {
}

bool Search_revamp::IsSearchActive() const {
	return false;
}
*/

void Search_revamp::Stop() {
}

/*
void Search_revamp::Abort() {
}
*/

namespace {
int indexOfHighestQEdge(Node_revamp* node) {
  float highestq = -2.0;
  int bestidx = -1;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    float q = node->GetEdges()[i].GetChild()->GetQ();
    if (q > highestq) {
      highestq = q;
      bestidx = i;
    }
  }
  return bestidx;
}
}

void Search_revamp::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }

}

/*
float Search_revamp::GetBestEval() const {
	return 1.0;
}

std::pair<Move, Move> Search_revamp::GetBestMove() const {
	return {Move("d2d4", false), NULL};
}

*/


Search_revamp::~Search_revamp() {
//  Abort();
  Wait();
}


//////////////////////////////////////////////////////////////////////////////
// Distribution
//////////////////////////////////////////////////////////////////////////////

std::vector<float> SearchWorker_revamp::q_to_prob(std::vector<float> Q, int depth, float multiplier, float max_focus) {
  bool DEBUG = false;
  // rebase depth from 0 to 1
  depth++;
    depth = sqrt(depth); // if we use full_tree_depth the distribution gets too sharp after a while
    auto min_max = std::minmax_element(std::begin(Q), std::end(Q));
    float min_q = Q[min_max.first-std::begin(Q)];
    float max_q = Q[min_max.second-std::begin(Q)];
    std::vector<float> q_prob (Q.size());    
    if(max_q - min_q == 0){
      // Return the uniform distribution
      for(int i = 0; i < (int)Q.size(); i++){
	q_prob[i] = 1.0f/Q.size();
      }
      assert(std::accumulate(q_prob.begin(), q_prob.end(), 0) == 1);      
      return(q_prob);
    }
    std::vector<float> a (Q.size());
    std::vector<float> b (Q.size());
    float c = 0;
    for(int i = 0; i < (int)Q.size(); i++){
      if(min_q < 0){
	// a[i] = (Q[i] - min_q) * (float)depth/(max_q - min_q);
	a[i] = (Q[i] - min_q)/(max_q - min_q);	
      } else {
	// a[i] = Q[i] * (float)depth/max_q;
	a[i] = Q[i]/max_q;
      }
      b[i] = exp(multiplier * a[i]);
    }
    std::for_each(b.begin(), b.end(), [&] (float f) {
    c += f;
});
    float my_sum = 0;
    for(int i = 0; i < (int)Q.size(); i++){
      q_prob[i] = b[i]/c;
      my_sum = my_sum + q_prob[i];
    }
    if(q_prob[min_max.second-std::begin(Q)] > max_focus){
      if(DEBUG) LOGFILE << "limiting p to max_focus because " << q_prob[min_max.second-std::begin(Q)] << " is more than " << max_focus << "\n";
      // find index of the second best.
      std::vector<float> q_prob_copy = q_prob;
      // Turn the second into the best by setting the max to the min - 1 in the copy to make sure it is now less than whatever the second best is.
      q_prob_copy[min_max.second-std::begin(Q)] = q_prob_copy[min_max.first-std::begin(Q)] - 1.0f;
      auto second_best = std::max_element(std::begin(q_prob_copy), std::end(q_prob_copy));
      q_prob[min_max.second-std::begin(Q)] = max_focus;
      if(DEBUG) LOGFILE << "Setting the second best (at: " << second_best-std::begin(q_prob_copy) << ") " << q_prob[second_best-std::begin(q_prob_copy)] << " to the 1 - max_focus: \n";
      q_prob[second_best-std::begin(q_prob_copy)] = 1 - max_focus;
      // Set all the others to zero
      for(int i = 0; i < (int)Q.size(); i++){
	if(i != min_max.second-std::begin(Q) && i != second_best-std::begin(q_prob_copy)){
	  q_prob[i] = 0;
	}
	if(DEBUG) LOGFILE << "i: " << i << " q_prob[i]: " << q_prob[i] << "\n";
      }
    }
    // Does it sum to 1?
    assert(std::accumulate(q_prob.begin(), q_prob.end(), 0) == 1); // not sure if this ever worked.
    if(abs(my_sum - 1) > 1e-5){ 
      LOGFILE << "sum of q_prob: " << std::setprecision(10) << my_sum << " largest q_prob: " << q_prob[min_max.second-std::begin(Q)] << " will change that to: " << q_prob[min_max.second-std::begin(Q)] - (my_sum - 1) << "\n";
      // Steal the extra from the best Q
      q_prob[min_max.second-std::begin(Q)] = q_prob[min_max.second-std::begin(Q)] - (my_sum - 1);
    }
    // Slightly less than 1 is fine, right? Otherwise uncomment this:
    // if(my_sum < 1.000000f){
    //   LOGFILE << "sum of q_prob: " << std::setprecision(10) << my_sum << " largest q_prob: " << q_prob[min_max.second-std::begin(Q)] << " will change that to: " << q_prob[min_max.second-std::begin(Q)] + (1 - my_sum) << "\n";
    //   // Steal the extra from the best Q
    //   q_prob[min_max.second-std::begin(Q)] = q_prob[min_max.second-std::begin(Q)] + (1 - my_sum);
    // }
    return(q_prob);
  }


float SearchWorker_revamp::computeChildWeights(Node_revamp* node) {
  switch (DISTRIBUTION_FUNCTION) {
    case 0:  // Hans'
    {

      if (MULTIPLE_NEW_SIBLINGS) {
        LOGFILE << "Hans' distribution is only implemented for non MULTIPLE_NEW_SIBLINGS alternative.";
        abort();
      }

// computes weights for the children based on average Qs (and possibly Ps) and, if there are unexpanded edges, a weight for the first unexpanded edge (the unexpanded with highest P)
// weights are >= 0, sum of weights is 1
// stored in weights_, idx corresponding to index in EdgeList

//  void SearchWorker_revamp::computeWeights(Node_revamp* node, int depth) {

  // compute depth because this value is not known otherwise
  // this can be optimised e.g. by storing depth in node
  int depth = 0;
  for (Node_revamp* nn = node; nn != worker_root_; nn = nn->GetParent()) {
    depth++;
  }

  double sum = 0.0;
  int n = node->GetNumChildren() + 1;
  if (n > node->GetNumEdges()) n = node->GetNumEdges();

  bool DEBUG = false;
  // if(depth == 0){
  //   DEBUG = true;
  // }

  // For debugging
  bool beta_to_move = (depth % 2 != 0);
  if(DEBUG) {
    LOGFILE << "Depth: " << depth << "\n";
  }

  // If no child is extended, then just use P. 
  if(n == 1 && (node->GetEdges())[0].GetChild() == nullptr){
    if(DEBUG) {
     LOGFILE << "No child extended yet, use P \n";
     float p = (node->GetEdges())[0].GetP();
     LOGFILE << "move: " << (node->GetEdges())[0].GetMove(beta_to_move).as_string() << " P: " << p << " \n";
    }
    return 0.0;
  } else {
    // At least one child is extended, weight by Q.

    std::vector<float> Q_prob (n);
    // float multiplier = 4.7f;
    float max_focus = 0.91f;
    auto board = history_.Last().GetBoard();
    int n_pieces_left = (board.ours() + board.theirs()).count();
    float a = 1.0f/1000.0f;
    float b = 1.0f/10.0f;
    float c = 0.8f;
    float multiplier = n_pieces_left * n_pieces_left * a + n_pieces_left * b + c;

    // float max_focus = 0.65 + 0.01 * n_pieces_left;
    std::vector<float> Q (n);

    // Populate the vector Q, all but the last child already has it.
    for (int i = 0; i < n-1; i++) {
      Q[i] = (node->GetEdges())[i].GetChild()->GetQ();
      if(DEBUG){
	float p = (node->GetEdges())[i].GetP();
	LOGFILE << "move: " << (node->GetEdges())[i].GetMove(beta_to_move).as_string() << " P: " << p << " Q: " << Q[i] << " \n";
      }
    }
    
    if((node->GetEdges())[n-1].GetChild() != nullptr){
      // All children have a Q value
      Q[n-1] = (node->GetEdges())[n-1].GetChild()->GetQ();
    } else {
      Q[n-1] = -0.05; // used in the rare case that q of better sibbling happens to be exactly 0.
      // Simplistic estimate: let the ratio between the P values be the ratio of the q values too.
      // If Q is below 0, then reverse the nominator and the denominator
      if((node->GetEdges())[n-2].GetChild()->GetQ() > 0) {
	Q[n-1] = (node->GetEdges())[n-2].GetChild()->GetQ() * (node->GetEdges())[n-1].GetP() / (node->GetEdges())[n-2].GetP();
      } 
      if((node->GetEdges())[n-2].GetChild()->GetQ() < 0) {
	Q[n-1] = (node->GetEdges())[n-2].GetChild()->GetQ() * (node->GetEdges())[n-2].GetP() / (node->GetEdges())[n-1].GetP();	  
      }
    }
    if(DEBUG){
      float p = (node->GetEdges())[n-1].GetP();
      LOGFILE << "move: " << (node->GetEdges())[n-1].GetMove(beta_to_move).as_string() << " P: " << p << " Q: " << Q[n-1];
      if((node->GetEdges())[n-1].GetChild() == nullptr){
	LOGFILE << " (estimated value) \n";
      } else {
	LOGFILE << " \n";
      }
    }

    if(DEBUG) LOGFILE << "calling q_to_prob()\n";
    Q_prob = q_to_prob(Q, full_tree_depth, multiplier, max_focus);
    if(DEBUG) LOGFILE << "q_to_prob() returned \n";    

    if((node->GetEdges())[n-1].GetChild() == nullptr){  // There is unexpanded edge
      n--;
    }

    for(int i = 0; i < n; i++){
      node->GetEdges()[i].GetChild()->SetW(Q_prob[i]);
      sum += Q_prob[i];
      if(DEBUG){
      	LOGFILE << "move: " << (node->GetEdges())[i].GetMove(beta_to_move).as_string() << " Q as prob: " << Q_prob[i] << "\n";
      }
    }
  }

  if(sum - 1 > 1e-5){
    LOGFILE << "sum: " << std::setprecision(6) << sum << "\n";
  }
  
  return sum;

  // // Probably not needed anymore, since q_to_prob returns a vector with sum=1 and else there is only one alternative: highest P.
  // if (sum > 0.0) {
  //   float scale = (float)(1.0 / sum);
  //   for (int i = 0; i < n; i++) {
  //     weights_[widx + i] *= scale;
  //   }
  // } else {
  //   float x = 1.0f / (float)n;
  //   for (int i = 0; i < n; i++) {
  //     weights_[widx + i] = x;
  //   }
  // }

//}


    } break;
    case 1:  // Plain exponential
    {
      int n = node->GetNumChildren();
      // LOGFILE << "number of children: " << n;
      if (n == 0) {
        return 0.0;
      } else {
        if (MULTIPLE_NEW_SIBLINGS) {
          double sum1 = 0.0;
          double sum2 = 0.0;
          for (int i = 0; i < n; i++) {
            float w = exp(q_concentration_ * node->GetEdges()[i].GetChild()->GetQ());
            node->GetEdges()[i].GetChild()->SetW(w);
            sum1 += w;
            sum2 += node->GetEdges()[i].GetP();
          }
          sum1 = sum2 / sum1;
          for (int i = 0; i < n; i++) {
            node->GetEdges()[i].GetChild()->SetW(node->GetEdges()[i].GetChild()->GetW() * sum1);
          }
          return sum2;
        } else {

	  if(node != worker_root_) {
	    // LOGFILE << "at compute child weights for move: " << node->GetParent()->GetEdges()[node->GetIndex()].GetMove(search_->played_history_.IsBlackToMove()).as_string();
	  }
	  // sum_of_w_of_expanded_nodes is the sum of the exponentiated Q:s (where Q is not OrigQ but a backpropagated Q)
	  double sum_of_P_of_expanded_nodes = 0.0;
	  double sum_of_w_of_expanded_nodes = 0.0;
          for (int i = 0; i < n; i++) {
            float w = exp(q_concentration_ * node->GetEdges()[i].GetChild()->GetQ());
            node->GetEdges()[i].GetChild()->SetW(w);
            sum_of_w_of_expanded_nodes += w;
	    sum_of_P_of_expanded_nodes += node->GetEdges()[i].GetP();
	    // LOGFILE << "Raw P of node: " << i << " is " << node->GetEdges()[i].GetP();
	    // LOGFILE << "Raw Q of node: " << i << " is " << node->GetEdges()[i].GetChild()->GetOrigQ();	    
          }
	  // factor for normalising w:s so their sum matches sum_of_P_of_expanded_nodes
	  double normalise_to_sum_of_p = sum_of_P_of_expanded_nodes / sum_of_w_of_expanded_nodes;
	  
	  for (int i = 0; i < n; i++) {
	    // LOGFILE << "normalising w for element: " << i << " from " << node->GetEdges()[i].GetChild()->GetW() << " to " << node->GetEdges()[i].GetChild()->GetW() * normalise_to_sum_of_p << " since sum_of_P_expanded_nodes is " << sum_of_P_of_expanded_nodes;
	    node->GetEdges()[i].GetChild()->SetW(node->GetEdges()[i].GetChild()->GetW() * normalise_to_sum_of_p );
	  }

	  // Critique of the chunk below: Whenever there are unexpanded edges, the sum of the P for the new child and the sum of the P for the already expanded nodes is less than one.
	  // The above gives the leftovers to the expanded nodes, but why should _they_ have it? That creates an imbalance in their favor compared to the new child.
	  // My intuition says its better to scale both the P of the new child and the sum of P for the already expanded nodes. 
	  // sum2 what is left over when the new P of the child has been subtracted from one.
	  
          // float sum2 = 1.0;
          // if (node->GetNumChildren() < node->GetNumEdges()) {
          //   sum2 -= node->GetEdges()[node->GetNumChildren()].GetP();
          // }
	  // // sum1 is now a weight used to scale up the exponentiated Q:s so that they sum up to sum2.
          // sum1 = sum2 / sum1;
          // for (int i = 0; i < n; i++) {
          //   node->GetEdges()[i].GetChild()->SetW(node->GetEdges()[i].GetChild()->GetW() * sum1);
          // }
          // // return sum2;

	  // We want to use P even if there is a Q available, since single Q values are always uncertain, but becomes more certain the more subnodes there is.
	  // In UCT the choice is made using the formula (highest score gets picked):
	  // Q + GetP() * cpuct * sqrt(max(GetN(), 1)) / (1 + GetN())
	  // In short, Policy is weighted by sqrt(N) / N (q in their formula is OrigQ)
	  // Looking the curve y = sqrt(x)/x, my guess it that decreases too quickly for chess, we need to explore more than that, or we will not find tactical tricks.
	  // (in UCT the constant cpuct is multiplied to the sqrt(x)/x to the get the final factor.
	  // Let's try x^0.7/x instead. At 100 visist the weight of policy would be 0.25, at 800 visists, the weight of policy would then be 0.13
	  // We can of course make this a run time parameter but, for now I'll use a constant for it.
	  
	  std::vector<double> weighted_p_and_q(n);
	  double sum_of_weighted_p_and_q = 0.0;
	  double p_weight_exponent = 0.7; 
	  for (int i = 0; i < n; i++){
	    double relative_weight_of_p = pow(node->GetEdges()[i].GetChild()->GetN(), p_weight_exponent) / ( 0.05 + node->GetEdges()[i].GetChild()->GetN()); // 0.05 is here to make Q have some influence after 1 visit.
	    // LOGFILE << "relative_weight_of_p:" << relative_weight_of_p;
	    double relative_weight_of_q = 1 - relative_weight_of_p;
	    // LOGFILE << "relative_weight_of_q:" << relative_weight_of_q;	    
	    if(relative_weight_of_q > 0){
	      weighted_p_and_q[i] = (relative_weight_of_q * node->GetEdges()[i].GetChild()->GetW() + relative_weight_of_p * node->GetEdges()[i].GetP()) ; 
	    } else {
	      weighted_p_and_q[i] = node->GetEdges()[i].GetP();
	    }
	    // LOGFILE << "Weighted p and q:" << weighted_p_and_q[i];
	    sum_of_weighted_p_and_q += weighted_p_and_q[i];
	  }
	  // make these sum to the sum of P of all the expanded children
	  double final_sum_of_weights_for_the_exanded_children = 0.0; // save the final sum here, we will return it.
	  for (int i = 0; i < n; i++){
	    node->GetEdges()[i].GetChild()->SetW(weighted_p_and_q[i] / sum_of_weighted_p_and_q * sum_of_P_of_expanded_nodes);
	    final_sum_of_weights_for_the_exanded_children += node->GetEdges()[i].GetChild()->GetW();
	    // LOGFILE << "visits: " << node->GetEdges()[i].GetChild()->GetN()
	    // 	    << " P: " << node->GetEdges()[i].GetP()
	    // 	    << " original Q: " << node->GetEdges()[i].GetChild()->GetOrigQ() 
	    // 	    << " Q after search: " << node->GetEdges()[i].GetChild()->GetQ()
	    // 	    << " q as prop: " << node->GetEdges()[i].GetChild()->GetW()
	    // 	    << " sum of p for the expanded nodes: " << sum_of_P_of_expanded_nodes
	    // 	    << " weighted sum of P and q_as_prob: " << weighted_p_and_q[i]
	    // 	    << " (weighted sum of P and q_as_prob) divided the product of sum_of_weighted_p_and_q and sum_of_P_of_expanded_nodes: " << node->GetEdges()[i].GetChild()->GetW();
	  }
	  if(node->GetNumChildren() < node->GetNumEdges()) {
	    // scale the p_q_weighted part so that it toghether with the P part sums to 1
	    double my_final_scaler = final_sum_of_weights_for_the_exanded_children + node->GetEdges()[node->GetNumChildren()].GetP();
	    final_sum_of_weights_for_the_exanded_children = 0.0; // recalc this one
	    for (int i = 0; i < n; i++){
	      node->GetEdges()[i].GetChild()->SetW(node->GetEdges()[i].GetChild()->GetW() / my_final_scaler);
	      final_sum_of_weights_for_the_exanded_children += node->GetEdges()[i].GetChild()->GetW();
	      // LOGFILE << "Final weight: " << node->GetEdges()[i].GetChild()->GetW();
	    }
	  } else {
	    // All children already expanded. This should already sum up to 1.
	    // LOGFILE << "All nodes already expanded, returning " << final_sum_of_weights_for_the_exanded_children;
	  }
	  return final_sum_of_weights_for_the_exanded_children;
        }
      }
    } break;
  }
}



//~ void printNodePos(Node_revamp* node, Node_revamp* root) {
  //~ while (true) {
    //~ if (node == root) break;
    //~ std::cerr << "-" << node->GetIndex();
    //~ node = node->GetParent();
  //~ }
  //~ std::cerr << ".";
//~ }



//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

// if queue is full, w must be larger than smallest weight
void SearchWorker_revamp::pushNewNodeCandidate(float w, Node_revamp* node, int idx) {
  if (node_prio_queue_.size() < (unsigned int)params_.GetMiniBatchSize()) {
    node_prio_queue_.push_back({w, node, idx});
    
    // bubble up
    int i = node_prio_queue_.size() - 1;
    while (i > 0) {
      int p = (i - 1)/2;
      if (node_prio_queue_[i].w < node_prio_queue_[p].w) {
        struct NewNodeCandidate tmp = node_prio_queue_[p];
        node_prio_queue_[p] = node_prio_queue_[i];
        node_prio_queue_[i] = tmp;
        i = p;
      } else {
        break;
      }
    }
  } else {
    node_prio_queue_[0] = {w, node, idx};
    
    // bubble down
    unsigned int i = 0;
    while (true) {
      if (2*i + 1 >= node_prio_queue_.size()) break;
      if (2*i + 2 >= node_prio_queue_.size()) {
        if (node_prio_queue_[2*i + 1].w < node_prio_queue_[i].w) {
          struct NewNodeCandidate tmp = node_prio_queue_[2*i + 1];
          node_prio_queue_[2*i + 1] = node_prio_queue_[i];
          node_prio_queue_[i] = tmp;
          i = 2*i + 1;
        }
        break;
      } else {
        if (node_prio_queue_[2*i + 1].w < node_prio_queue_[i].w && node_prio_queue_[2*i + 1].w <= node_prio_queue_[2*i + 2].w) {
          struct NewNodeCandidate tmp = node_prio_queue_[2*i + 1];
          node_prio_queue_[2*i + 1] = node_prio_queue_[i];
          node_prio_queue_[i] = tmp;
          i = 2*i + 1;
        } else {
          if (node_prio_queue_[2*i + 2].w < node_prio_queue_[i].w) {
            struct NewNodeCandidate tmp = node_prio_queue_[2*i + 2];
            node_prio_queue_[2*i + 2] = node_prio_queue_[i];
            node_prio_queue_[i] = tmp;
            i = 2*i + 2;
          } else {
            break;
          }
        }
      }
    }
  }

//  for (int i = 1; i < node_prio_queue_.size(); i++) {
//    if (node_prio_queue_[i].w < node_prio_queue_[(i - 1)/2].w) {
//      std::cerr << "ALERT!!!!!!!!!\n";
//    }
//  }
}

void SearchWorker_revamp::pickNodesToExtend(Node_revamp* node, float global_weight) {
  nodestack_.push_back(node);

  if (MULTIPLE_NEW_SIBLINGS) {

    float smallest_weight_in_queue = -1.0;
    if (node_prio_queue_.size() == (unsigned int)params_.GetMiniBatchSize()) {
      smallest_weight_in_queue = node_prio_queue_[0].w;
    }
    float oldw = 2.0;
    for (int i = node->GetNumChildren(); i < node->GetNumEdges(); i++) {
      float w = global_weight * node->GetEdges()[i].GetP();
      if (w > smallest_weight_in_queue) {
        while (w >= oldw) {
          w = nextafterf(w, -1.0);
        }
        pushNewNodeCandidate(w, node, i);
        if (node_prio_queue_.size() == (unsigned int)params_.GetMiniBatchSize()) {
          smallest_weight_in_queue = node_prio_queue_[0].w;
        }
        oldw = w;
      } else {
        break;
      }
    }
    
    for (int j = 0; j < node->GetNumChildren(); j++) {
      Node_revamp* child = node->GetEdges()[j].GetChild();
      if (child->GetNExtendable() > 0) {
        float w = global_weight * child->GetW();
        if (w * child->GetMaxW() > smallest_weight_in_queue) {
          pickNodesToExtend(child, w);
          if (node_prio_queue_.size() == (unsigned int)params_.GetMiniBatchSize()) {
            smallest_weight_in_queue = node_prio_queue_[0].w;
          }
        }
      }
    }
  } else {  // not MULTIPLE_NEW_SIBLINGS

    float smallest_weight_in_queue = -1.0;
    if (node_prio_queue_.size() == (unsigned int)params_.GetMiniBatchSize()) {
      smallest_weight_in_queue = node_prio_queue_[0].w;
    }

    float totw = 0.0;

    for (int j = 0; j < node->GetNumChildren(); j++) {
      Node_revamp* child = node->GetEdges()[j].GetChild();
      float w = child->GetW();
      totw += w;
      if (child->GetNExtendable() > 0) {
        w *= global_weight;
        if (w * child->GetMaxW() > smallest_weight_in_queue) {
          pickNodesToExtend(child, w);
          if (node_prio_queue_.size() == (unsigned int)params_.GetMiniBatchSize()) {
            smallest_weight_in_queue = node_prio_queue_[0].w;
          }
        }
      }
    }

    if (node->GetNumChildren() < node->GetNumEdges()) {
      totw = (1.0 - totw) * global_weight;
      if (totw > smallest_weight_in_queue) {
        pushNewNodeCandidate(totw, node, node->GetNumChildren());
      }
    }
  }
}
  


void SearchWorker_revamp::retrieveNNResult(Node_revamp* node, int batchidx) {
  float q = -computation_->GetQVal(batchidx);
  if (q < -1.0 || q > 1.0) {
    LOGFILE << "q = " << q;
    if (q < -1.0) q = -1.0;
    if (q > 1.0) q = 1.0;
  }
  node->SetOrigQ(q);

  float total = 0.0;
  int nedge = node->GetNumEdges();
  pvals_.clear();
  for (int k = 0; k < nedge; k++) {
    float p = computation_->GetPVal(batchidx, (node->GetEdges())[k].GetMove().as_nn_index());
    if (p < 0.0) {
      LOGFILE << "p value < 0\n";
      p = 0.0;
    }
    if (p_concentration_ != 1.0) {
      p = pow(p, p_concentration_);
    }
    pvals_.push_back(p);
    total += p;
  }
  if (total > 0.0f) {
    float scale = 1.0f / total;
    for (int k = 0; k < nedge; k++) {
      (node->GetEdges())[k].SetP(pvals_[k] * scale);
    }
    node->SortEdgesByPValue();
  } else {
    float x = 1.0f / (float)nedge;
    for (int k = 0; k < nedge; k++) {
      (node->GetEdges())[k].SetP(x);
    }
  }
  if (INITIAL_MAX_P_IS_1) {
    node->SetMaxW(1.0);
  } else {
    node->SetMaxW(node->GetEdges()[0].GetP());
  }
}

void SearchWorker_revamp::recalcPropagatedQ(Node_revamp* node) {
  bool DEBUG = false;
  float total_children_weight = computeChildWeights(node);

  if (total_children_weight < 0.0 || total_children_weight - 1.0 > 1.00012) {
    std::cerr << "total_children_weight: " << total_children_weight << "\n";
    abort();
  }
  float totw = 0.0;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    float w = node->GetEdges()[i].GetChild()->GetW();
    if (w < 0.0) {
      std::cerr << "w: " << w << "\n";
      abort();
    }
    totw += w;
  }
  if (abs(total_children_weight - totw) > 1.00012) {
    std::cerr << "total_children_weight: " << total_children_weight << ", totw: " << total_children_weight << "\n";
    abort();
  }

  // Average Q START
  // float q = (1.0 - total_children_weight) * node->GetOrigQ();
  // for (int i = 0; i < node->GetNumChildren(); i++) {
  //   q -= node->GetEdges()[i].GetChild()->GetW() * node->GetEdges()[i].GetChild()->GetQ();
  // }
  // node->SetQ(q);
  // Average Q STOP

  // Best guaranteed Q START
    // Current Q should be set to the inverse of Q of the child with the _highest_ Q.
    // TODO Don't update Q if the new value would be the same as the old.
    // Change Q only if: a new child has the highest Q, or the child that previously had the highest Q has a new Q.
    // For now, do it quick'n'dirty: change all nodes, even it the new value is the same as the old value
  std::vector<float> q_of_children (node->GetNumChildren());
  for (int i = 0; i < node->GetNumChildren(); i++) {
    if(node->GetEdges()[i].GetChild()->GetNumChildren() == 0){
      q_of_children[i] = node->GetEdges()[i].GetChild()->GetOrigQ();
    } else {
      q_of_children[i] = node->GetEdges()[i].GetChild()->GetQ();
    }
  }
  auto max = std::max_element(std::begin(q_of_children), std::end(q_of_children));
  float max_q = q_of_children[max-std::begin(q_of_children)];
  if(-max_q != node->GetQ()){
    node->SetQ(-max_q);
    if(DEBUG) { LOGFILE << "Update Q to " << -max_q << " for node " << node->GetParent()->GetEdges()[node->GetIndex()].GetMove(search_->played_history_.IsBlackToMove()).as_string(); }
  }
  // Best guaranteed Q STOP
  
  int n = 1;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    n += node->GetEdges()[i].GetChild()->GetN();
  }
  node->SetN(n);

  if (MULTIPLE_NEW_SIBLINGS)
    n = node->GetNumEdges() - node->GetNumChildren();
  else
    n = node->GetNumEdges() > node->GetNumChildren() ? 1 : 0;

  for (int i = 0; i < node->GetNumChildren(); i++) {
    n += node->GetEdges()[i].GetChild()->GetNExtendable();
  }
  node->SetNExtendable(n);

  float max_w = node->GetNumChildren() < node->GetNumEdges() ? node->GetEdges()[node->GetNumChildren()].GetP() : 0.0;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    float br_max_w = node->GetEdges()[i].GetChild()->GetW() * node->GetEdges()[i].GetChild()->GetMaxW();
    if (br_max_w > max_w) max_w = br_max_w;
  }
  node->SetMaxW(max_w);
}

int SearchWorker_revamp::appendHistoryFromTo(Node_revamp* from, Node_revamp* to) {
  movestack_.clear();
  while (to != from) {
    movestack_.push_back(to->GetParent()->GetEdges()[to->GetIndex()].GetMove());
    to = to->GetParent();
  }
  for (int i = movestack_.size() - 1; i >= 0; i--) {
    history_.Append(movestack_[i]);
  }
  return movestack_.size();
}

void SearchWorker_revamp::RunBlocking() {
  bool DEBUG = false;
  LOGFILE << "Running thread for node " << worker_root_ << "\n";
  auto board = history_.Last().GetBoard();
  if (DEBUG) LOGFILE << "Inital board:\n" << board.DebugString();

  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

  unsigned int lim = search_->limits_.visits;
  int i = 0;

  if (worker_root_->GetNumEdges() == 0 && !worker_root_->IsTerminal()) {  // root node not extended
    worker_root_->ExtendNode(&history_, MULTIPLE_NEW_SIBLINGS, worker_root_);
    if (worker_root_->IsTerminal()) {
      LOGFILE << "Root " << worker_root_ << " is terminal, nothing to do\n";
      return;
    }
    minibatch_.clear();
    computation_ = search_->network_->NewComputation();
    AddNodeToComputation();

    // LOGFILE << "Computing thread root ..";
    computation_->ComputeBlocking();
    // LOGFILE << " done\n";
    retrieveNNResult(worker_root_, 0);
    i++;
  }

  int64_t last_uci_time = 0;

  while (worker_root_->GetN() < lim) {
    minibatch_.clear();
    computation_ = search_->network_->NewComputation();

    pickNodesToExtend(worker_root_, 1.0);

    LOGFILE << "n: " << worker_root_->GetN()
            << ", n_extendable: " << worker_root_->GetNExtendable()
            << ", queue size: " << node_prio_queue_.size()
            << ", lowest w: " << node_prio_queue_[0].w
            << ", node stack size: " << nodestack_.size()
            << ", max_unexpanded_w: " << worker_root_->GetMaxW();

    for (unsigned int i = 0; i < node_prio_queue_.size(); i++) {
      Node_revamp* node = node_prio_queue_[i].node;
      int idx = node_prio_queue_[i].idx;

      int nappends = appendHistoryFromTo(worker_root_, node);
      node->GetEdges()[idx].CreateChild(node, idx, nappends + 1);
      Node_revamp* newchild = node->GetEdges()[idx].GetChild();

      history_.Append(node->GetEdges()[idx].GetMove());

      newchild->ExtendNode(&history_, MULTIPLE_NEW_SIBLINGS, worker_root_);
      if (!newchild->IsTerminal()) {
        AddNodeToComputation();
        minibatch_.push_back(newchild);
      } 

      for (int j = 0; j <= nappends; j++) {
        history_.Pop();
      }
      
      if (nappends > full_tree_depth) full_tree_depth = nappends;
      cum_depth_ += nappends;
    }

    node_prio_queue_.clear();

    if(minibatch_.size() == 0){
      // E.g. child from root with best policy is terminal
      LOGFILE << "Couldn't find any move to send to eval.";

      for (int n = nodestack_.size(); n > 0; n--) {
	Node_revamp* node = nodestack_.back();
	nodestack_.pop_back();
	if(node->GetNumChildren() > 0){
	  recalcPropagatedQ(node);
	}
      }
       
      break;
    } else {
      LOGFILE << "Computing batch of size " << minibatch_.size();

    //~ std::this_thread::sleep_for(std::chrono::milliseconds(0));
    //~ LOGFILE << "RunNNComputation START ";
    //~ start_comp_time_ = std::chrono::steady_clock::now();

      computation_->ComputeBlocking();

    // stop_comp_time_ = std::chrono::steady_clock::now();
    // auto duration = stop_comp_time_ - start_comp_time_;
    // LOGFILE << "RunNNComputation STOP nanoseconds used: " << duration.count() << "; ";
    // int idx_in_computation = minibatch_.size();
    // int duration_mu = duration.count();
    // if(duration_mu > 0){
    //   float better_duration = duration_mu / 1000;
    //   float nps = 1000 * idx_in_computation / better_duration;
    //   LOGFILE << " nodes in last batch that were evaluated " << idx_in_computation << " nps " << 1000 * nps << "\n";
    // }
    
    
      i += minibatch_.size();
    
      for (int j = 0; j < (int)minibatch_.size(); j++) {
	retrieveNNResult(minibatch_[j], j);
      }

      for (int n = nodestack_.size(); n > 0; n--) {
	Node_revamp* node = nodestack_.back();
	nodestack_.pop_back();
	if(node->GetNumChildren() > 0){
	  recalcPropagatedQ(node);
	}
      }
    
      int64_t time = search_->GetTimeSinceStart();
      if (time - last_uci_time > kUciInfoMinimumFrequencyMs) {
	last_uci_time = time;
	SendUciInfo();
      }
    }
  }

  int64_t elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
  LOGFILE << "Elapsed time when thread for node " << worker_root_ << " which has size " << worker_root_->GetN() << " nodes did " << i << " computations: " << elapsed_time << "ms";

  LOGFILE << "root Q: " << worker_root_->GetQ();

  LOGFILE << "move   P                 n   norm n      h   Q          w";
  for (int i = 0; i < worker_root_->GetNumChildren(); i++) {
    LOGFILE << std::fixed << std::setfill(' ') 
              << (worker_root_->GetEdges())[i].GetMove().as_string() << " "
              << std::setw(10) << (worker_root_->GetEdges())[i].GetP() << " "
              << std::setw(10) << (worker_root_->GetEdges())[i].GetChild()->GetN() << " "
              << std::setw(10) << (float)(worker_root_->GetEdges())[i].GetChild()->GetN() / (float)(worker_root_->GetN() - 1) << " "
              << std::setw(4) << (worker_root_->GetEdges())[i].GetChild()->ComputeHeight() << " "
              << std::setw(10) << (float)(worker_root_->GetEdges())[i].GetChild()->GetQ() << " "
              << std::setw(10) << worker_root_->GetEdges()[i].GetChild()->GetW();
  }

  int bestidx = indexOfHighestQEdge(search_->root_node_);
  Move best_move = search_->root_node_->GetEdges()[bestidx].GetMove(search_->played_history_.IsBlackToMove());
  int ponderidx = indexOfHighestQEdge(search_->root_node_->GetEdges()[bestidx].GetChild());
  // If the move we make is terminal, then there is nothing to ponder about
  if(!search_->root_node_->GetEdges()[bestidx].GetChild()->IsTerminal()){
    Move ponder_move = search_->root_node_->GetEdges()[bestidx].GetChild()->GetEdges()[ponderidx].GetMove(!search_->played_history_.IsBlackToMove());
    search_->best_move_callback_({best_move, ponder_move});    
  } else {
    search_->best_move_callback_({best_move});    
  }
}

void SearchWorker_revamp::AddNodeToComputation() {
  // auto hash = history_.HashLast(params_.GetCacheHistoryLength() + 1);
  auto planes = EncodePositionForNN(history_, 8, params_.GetHistoryFill());
  // std::vector<uint16_t> moves;
  // int nedge = node->GetNumEdges();
  // for (int k = 0; k < nedge; k++) {
  //   moves.emplace_back(node->edges_[k].GetMove().as_nn_index());
  // }
  computation_->AddInput(std::move(planes));
  //computation_->AddInput(hash, std::move(planes), std::move(moves));
}


void SearchWorker_revamp::SendUciInfo() {

  auto score_type = params_.GetScoreType();

  ThinkingInfo common_info;
  if (worker_root_->GetN() > search_->initial_visits_)
    common_info.depth = cum_depth_ / (worker_root_->GetN() - search_->initial_visits_);
  common_info.seldepth = full_tree_depth;
  common_info.time = search_->GetTimeSinceStart();
  common_info.nodes = worker_root_->GetN();
  common_info.nps =
      common_info.time ? ((worker_root_->GetN() - search_->initial_visits_) * 1000 / common_info.time) : 0;

  std::vector<ThinkingInfo> uci_infos;

  int multipv = 0;

  float prevq = 2.0;
  int previdx = -1;
  for (int i = 0; i < worker_root_->GetNumChildren(); i++) {  
    float bestq = -2.0;
    int bestidx = -1;
    for (int j = 0; j < worker_root_->GetNumChildren(); j++) {
      float q = worker_root_->GetEdges()[j].GetChild()->GetQ();
      if (q > bestq && (q < prevq || (q == prevq && j > previdx))) {
        bestq = q;
        bestidx = j;
      }
    }
    prevq = bestq;
    previdx = bestidx;

    ++multipv;

    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();

    if (score_type == "centipawn") {
      uci_info.score = 290.680623072 * tan(1.548090806 * bestq);
    } else if (score_type == "win_percentage") {
      uci_info.score = bestq * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = bestq * 10000;
    }

    if (params_.GetMultiPv() > 1) uci_info.multipv = multipv;
    bool flip = history_.IsBlackToMove();
    uci_info.pv.push_back(worker_root_->GetEdges()[bestidx].GetMove(flip));
    Node_revamp* n = worker_root_->GetEdges()[bestidx].GetChild();
    while (n && n->GetNumChildren() > 0) {
      flip = !flip;
      int bestidx = indexOfHighestQEdge(n);
      uci_info.pv.push_back(n->GetEdges()[bestidx].GetMove(flip));
      n = n->GetEdges()[bestidx].GetChild();
    }
  }

  // reverse the order
  std::reverse(uci_infos.begin(), uci_infos.end());
  search_->info_callback_(uci_infos);

}



}  // namespace lczero
