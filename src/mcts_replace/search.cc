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
      //~ start_time_(std::chrono::steady_clock::now()),
      //~ initial_visits_(root_node_->GetN()),
      best_move_callback_(best_move_callback)
      //~ info_callback_(info_callback),
    {}


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

float SearchWorker_revamp::computeChildWeights(Node_revamp* node) {
  int n = node->GetNumChildren();
  
  if (n == 0) {
    return 0.0;
  } else {
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
    float w = global_weight * child->GetW();
    if (w * child->GetMaxW() > smallest_weight_in_queue) {
      history_.Append(node->GetEdges()[j].GetMove());
      pickNodesToExtend(child, w);
      history_.Pop();
      if (node_prio_queue_.size() == (unsigned int)params_.GetMiniBatchSize()) {
        smallest_weight_in_queue = node_prio_queue_[0].w;
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
  node->SetMaxW(node->GetEdges()[0].GetP());
}

void SearchWorker_revamp::recalcPropagatedQ(Node_revamp* node) {
  float total_children_weight = computeChildWeights(node);
  
  float q = (1.0 - total_children_weight) * node->GetOrigQ();
  for (int i = 0; i < node->GetNumChildren(); i++) {
    q -= node->GetEdges()[i].GetChild()->GetW() * node->GetEdges()[i].GetChild()->GetQ();
  }
  node->SetQ(q);
  
  int n = 1;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    n += node->GetEdges()[i].GetChild()->GetN();
  }
  node->SetN(n);

  n = node->GetNumEdges() - node->GetNumChildren();
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
  LOGFILE << "Running thread for node " << worker_root_ << "\n";
  auto board = history_.Last().GetBoard();
  LOGFILE << "Inital board:\n" << board.DebugString();

  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

  unsigned int lim = search_->limits_.visits;
  int i = 0;

  if (worker_root_->GetNumEdges() == 0 && !worker_root_->IsTerminal()) {  // root node not extended
    worker_root_->ExtendNode(&history_);
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
      
      node->GetEdges()[idx].CreateChild(node, idx);
      Node_revamp* newchild = node->GetEdges()[idx].GetChild();

      int nappends = appendHistoryFromTo(worker_root_, node);
      history_.Append(node->GetEdges()[idx].GetMove());

      newchild->ExtendNode(&history_);
      if (!newchild->IsTerminal()) {
        AddNodeToComputation();
        minibatch_.push_back(newchild);
      } else {
        // LOGFILE << "Terminal node created\n";	
      }

      for (int j = 0; j <= nappends; j++) {
        history_.Pop();
      }

    }

    node_prio_queue_.clear();
    
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
      recalcPropagatedQ(node);
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
  Move best_move = search_->root_node_->GetEdges()[bestidx].GetMove();
  int ponderidx = indexOfHighestQEdge(search_->root_node_->GetEdges()[bestidx].GetChild());
  // Move ponder_move = search_->root_node_->GetEdges()[bestidx].GetChild()->GetEdges()[ponderidx].GetMove(true);
  // When we are to play black this fails, it returns the move from whites perspective.
  Move ponder_move = search_->root_node_->GetEdges()[bestidx].GetChild()->GetEdges()[ponderidx].GetMove(!history_.IsBlackToMove());
  search_->best_move_callback_({best_move, ponder_move});
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




}  // namespace lczero
