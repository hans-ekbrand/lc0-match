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

#include "mcts_replace/search.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>

#include "neural/encoder.h"

namespace lczero {

namespace {

// Alternatives:

int const MAX_NEW_SIBLINGS = 10000;
  // The maximum number of new siblings. If 1, then it's like old MULTIPLE_NEW_SIBLINGS = false, if >= maximum_number_of_legal_moves it's like MULTIPLE_NEW_SIBLINGS = true
const int kUciInfoMinimumFrequencyMs = 500;

int const N_HELPER_THREADS = 2;

bool const LOG_RUNNING_INFO = false;

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


//////////////////////////////////////////////////////////////////////////////
// Search
//////////////////////////////////////////////////////////////////////////////

Search_revamp::Search_revamp(const NodeTree_revamp& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits_revamp& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb,
               bool ponder)
    : root_node_(tree.GetCurrentHead()),
      cache_(cache),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      limits_(limits),
      params_(options),
      ponder_(ponder),
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

int64_t Search_revamp::GetTimeToDeadline() const {
  if (!limits_.search_deadline) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             *limits_.search_deadline - std::chrono::steady_clock::now())
      .count();
}


void Search_revamp::StartThreads(size_t how_many) {
	threads_list_mutex_.lock();
	for (int i = threads_.size(); i < (int)how_many; i++) {
		n_thread_active_++;
		threads_.emplace_back([this, i]()
			{
				SearchWorker_revamp worker(this);
				worker.ThreadLoop(i);
			}
		);
	}
	threads_list_mutex_.unlock();
}

void Search_revamp::RunBlocking(size_t threads) {
	StartThreads(threads);
	Wait();
}

void Search_revamp::Stop() {
  // If pondering is on, then turn if off to get bestmove
  ponder_ = false;
  not_stop_searching_ = false;
  // TODO: If search is already finished, then return bestmove
}

void Search_revamp::Abort() {
	abort_ = true;
	not_stop_searching_ = false;
}

bool Search_revamp::IsSearchActive() const {
	//threads_list_mutex_.lock();
	bool active = n_thread_active_ > 0;
	//threads_list_mutex_.unlock();
	return active;
}

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
	threads_list_mutex_.lock();
	while (!threads_.empty()) {
		threads_.back().join();
		threads_.pop_back();
	}
	threads_list_mutex_.unlock();
}


Search_revamp::~Search_revamp() {
	Abort();
	Wait();
}

void Search_revamp::SendUciInfo() {

  auto score_type = params_.GetScoreType();

  ThinkingInfo common_info;
  if (root_node_->GetN() > initial_visits_)
    common_info.depth = cum_depth_ / (root_node_->GetN() - initial_visits_);
  common_info.seldepth = full_tree_depth_;
  common_info.time = GetTimeSinceStart();
  common_info.nodes = root_node_->GetN();
  common_info.nps =
      common_info.time ? ((root_node_->GetN() - initial_visits_) * 1000 / common_info.time) : 0;

  std::vector<ThinkingInfo> uci_infos;

  int multipv = 0;

  float prevq = 2.0;
  int previdx = -1;
  for (int i = 0; i < root_node_->GetNumChildren(); i++) {  
    float bestq = -2.0;
    int bestidx = -1;
    for (int j = 0; j < root_node_->GetNumChildren(); j++) {
      float q = root_node_->GetEdges()[j].GetChild()->GetQ();
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
    bool flip = played_history_.IsBlackToMove();
    uci_info.pv.push_back(root_node_->GetEdges()[bestidx].GetMove(flip));
    Node_revamp* n = root_node_->GetEdges()[bestidx].GetChild();
    while (n && n->GetNumChildren() > 0) {
      flip = !flip;
      int bestidx = indexOfHighestQEdge(n);
      uci_info.pv.push_back(n->GetEdges()[bestidx].GetMove(flip));
      n = n->GetEdges()[bestidx].GetChild();
    }
  }

  // reverse the order
  std::reverse(uci_infos.begin(), uci_infos.end());
  info_callback_(uci_infos);

}

void Search_revamp::checkLimitsAndMaybeTriggerStop() {
	//root_node_->GetN() + (search_->n_thread_active_ - 1) * batch_size_ < visits/* && root_node_->GetNExtendable() > 0*/
	if (limits_.playouts >= 0 && root_node_->GetN() - initial_visits_ >= limits_.playouts) {
		not_stop_searching_ = false;
	} else
	if (limits_.visits >= 0 && root_node_->GetN() >= limits_.visits) {
		not_stop_searching_ = false;
	} else
	if (limits_.search_deadline && GetTimeToDeadline() <= 0) {
		not_stop_searching_ = false;
	} else
	if (limits_.depth >= 0 && cum_depth_ / (root_node_->GetN() - initial_visits_) >= (uint64_t)limits_.depth) {
		not_stop_searching_ = false;
	}
}

std::vector<std::string> Search_revamp::GetVerboseStats(Node_revamp* node, bool is_black_to_move) {

  std::vector<std::string> infos;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    std::ostringstream oss;
    oss << std::fixed;

    oss << std::left << std::setw(5)
        << node->GetEdges()[i].GetMove(is_black_to_move).as_string();

    oss << " (" << std::setw(4) << node->GetEdges()[i].GetMove(is_black_to_move).as_nn_index() << ")";

    oss << " N: " << std::right << std::setw(7) << node->GetEdges()[i].GetChild()->GetN() << " (+"
        << std::setw(2) << node->GetEdges()[i].GetChild()->GetN() << ") ";

    oss << "(P: " << std::setw(5) << std::setprecision(2) << node->GetEdges()[i].GetP() * 100
        << "%) ";

    oss << "(Q: " << std::setw(8) << std::setprecision(5) << node->GetEdges()[i].GetChild()->GetQ()
        << ") ";

    oss << "(U: " << std::setw(6) << std::setprecision(5) << node->GetEdges()[i].GetChild()->GetQ()
        << ") ";

    oss << "(Q+U: " << std::setw(8) << std::setprecision(5)
        << node->GetEdges()[i].GetChild()->GetQ() + node->GetEdges()[i].GetChild()->GetQ() << ") ";

    oss << "(V: ";
    optional<float> v;
    if (node->GetEdges()[i].GetChild()->IsTerminal()) {
      v = node->GetEdges()[i].GetChild()->GetQ();
    } else {
      v = node->GetEdges()[i].GetChild()->GetQ();
    }
    if (v) {
      oss << std::setw(7) << std::setprecision(4) << *v;
    } else {
      oss << " -.----";
    }
    oss << ") ";

    if (node->GetEdges()[i].GetChild()->IsTerminal()) oss << "(T) ";
    infos.emplace_back(oss.str());
  }
  return infos;
}

// void Search_revamp::SendMovesStats() {
//   const bool is_black_to_move = played_history_.IsBlackToMove();
//   auto move_stats = SearchWorker_revamp::GetVerboseStats(root_node_, is_black_to_move);

//   if (params_.GetVerboseStats()) {
//     // LOGFILE << "captured GetVerboseStats";
//     std::vector<ThinkingInfo> infos;
//     std::transform(move_stats.begin(), move_stats.end(),
//                    std::back_inserter(infos), [](const std::string& line) {
//                      ThinkingInfo info;
//                      info.comment = line;
//                      return info;
//                    });
//     info_callback_(infos);
//   } else {
//     LOGFILE << "=== Move stats:";
//     for (const auto& line : move_stats) LOGFILE << line;
//   }
// }

//////////////////////////////////////////////////////////////////////////////
// Distribution
//////////////////////////////////////////////////////////////////////////////

float SearchWorker_revamp::computeChildWeights(Node_revamp* node) {
  int n = node->GetNumChildren();
  bool DEBUG = false;
  // If no child is extended, then just use P.
  if(n == 0 && (node->GetEdges())[0].GetChild() == nullptr){
    if(DEBUG) LOGFILE << "No child extended, returning 0";
    return 0.0;
  } else {
    if(DEBUG) LOGFILE << "At computeChildWeights, number of (expanded) children: " << n;
    // At least one child is extended, weight the expanded children by Q.
    // sum_of_w_of_expanded_nodes is the sum of the exponentiated Q:s (where Q is not OrigQ but a backpropagated Q)
    double sum_of_P_of_expanded_nodes = 0.0;
    double sum_of_w_of_expanded_nodes = 0.0;
    for (int i = 0; i < n; i++) {
      // float w = exp(q_concentration_ * node->GetEdges()[i].GetChild()->GetQ());
      float w = exp(10 * pow(node->GetN(), 0.09) * node->GetEdges()[i].GetChild()->GetQ());
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
    for (int i = 0; i < n; i++){
      // double relative_weight_of_p = pow(node->GetEdges()[i].GetChild()->GetN(), policy_weight_exponent_) / ( 0.05 + node->GetEdges()[i].GetChild()->GetN()); // 0.05 is here to make Q have some influence after 1 visit.
      double relative_weight_of_p = pow(node->GetN(), (0.46 + log10(node->GetN())/30)) / ( 0.05 + node->GetN());
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
    return final_sum_of_weights_for_the_exanded_children;  // this should be same as sum_of_P_of_expanded_nodes
  }
}


void SearchWorker_revamp::pickNodesToExtend() {
	Node_revamp* node;
	int best_idx;

//	int nodes_visited = 0;

	for (int n_left = batch_size_; n_left > 0; n_left--) {
		node = root_node_;

		while (true) {
//			nodes_visited++;
			best_idx = node->GetBestIdx();
			if (best_idx == -1) {
				int nidx = node->GetNextUnexpandedEdge();
				if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
					node->GetEdges()[nidx].CreateChild(node, nidx);
					//new_nodes_list_lock_.lock();
					//new_nodes_.push_back({node, nidx, 0});
					//new_nodes_list_lock_.unlock();
					new_nodes_[new_nodes_size_] = {node->GetEdges()[nidx].GetChild(), 0, -1};
					new_nodes_size_++;
					node->SetNextUnexpandedEdge(nidx + 1);
					break;
				} else {  // no more child to add (before retrieved information about previous ones)
					return;
				}
			} else {
				node = node->GetEdges()[best_idx].GetChild();
			}
		}

		bool update_branching = true;

		while (true) {
			int16_t max_idx = -1;
			float max_w = 0.0;
			int nidx = node->GetNextUnexpandedEdge();
			if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
				max_w = node->GetEdges()[nidx].GetP();
			}
			for (int i = 0; i < node->GetNumChildren(); i++) {
				float br_max_w = node->GetEdges()[i].GetChild()->GetW() * node->GetEdges()[i].GetChild()->GetMaxW();
				if (br_max_w > max_w) {
					max_w = br_max_w;
					max_idx = i;
				}
			}
			node->SetMaxW(max_w);
			node->SetBestIdx(max_idx);

			if (update_branching) {
				uint8_t n = node->GetBranchingInFlight();
				if (n == 0) {
					node->SetBranchingInFlight(1);
				} else {
					node->SetBranchingInFlight(n + 1);
					update_branching = false;
				}
			}

			if (node == root_node_) break;
			node = node->GetParent();
		}
	}

//	LOGFILE << "picknodestoextend, nodes visited: " << nodes_visited;

}

void SearchWorker_revamp::buildJunctionRTree() {
//	for (int i = new_nodes_.size() - 1; i >= 0; i--) {
	for (int i = new_nodes_size_ - 1; i >= 0; i--) {
		Node_revamp* node = new_nodes_[i].node->GetParent();
		uint16_t* parent_ptr = &new_nodes_[i].junction;
		while (true) {
			while (node->GetBranchingInFlight() == 1) {
				if (node == root_node_) {
					*parent_ptr = 0xFFFF;
					break;
				}
				node = node->GetParent();
			}
			if (*parent_ptr == 0xFFFF) break;
			uint16_t &junc_idx = junction_of_node_[node];
			if (junc_idx == 0) {
				uint16_t new_junc_idx = junctions_.size();
				*parent_ptr = new_junc_idx;
				junctions_.push_back({node, 0, node->GetBranchingInFlight()});
				junc_idx = new_junc_idx + 1;
				parent_ptr = &junctions_[new_junc_idx].parent;
				if (node == root_node_) {
					*parent_ptr = 0xFFFF;
					break;
				}
				node = node->GetParent();
			} else {
				*parent_ptr = junc_idx - 1;
				break;
			}
		}
	}

//	for (int i = new_nodes_.size() - 1; i >= 0; i--) {
	for (int i = new_nodes_size_ - 1; i >= 0; i--) {
		Node_revamp* node = new_nodes_[i].node->GetParent();
		while (node->GetBranchingInFlight() > 0) {
			node->SetBranchingInFlight(0);
			if (node == root_node_) break;
			node = node->GetParent();
		}
	}
}

int SearchWorker_revamp::appendHistoryFromTo(std::vector<Move> *movestack, PositionHistory *history, Node_revamp* from, Node_revamp* to) {
  movestack->clear();
  while (to != from) {
    movestack->push_back(to->GetParent()->GetEdges()[to->GetIndex()].move_);
    to = to->GetParent();
  }
  for (int i = movestack->size() - 1; i >= 0; i--) {
    history->Append((*movestack)[i]);
  }
  return movestack->size();
}


void SearchWorker_revamp::AddNodeToComputation(PositionHistory *history) {
  // auto hash = history_.HashLast(params_.GetCacheHistoryLength() + 1);
  auto planes = EncodePositionForNN(*history, 8, history_fill_);
  // std::vector<uint16_t> moves;
  // int nedge = node->GetNumEdges();
  // for (int k = 0; k < nedge; k++) {
  //   moves.emplace_back(node->edges_[k].GetMove().as_nn_index());
  // }
  computation_->AddInput(std::move(planes));
  //computation->AddInput(hash, std::move(planes), std::move(moves));
}


int SearchWorker_revamp::extendTree(std::vector<Move> *movestack, PositionHistory *history) {
	int count = 0;

	int full_tree_depth = search_->full_tree_depth_;
	int cum_depth = 0;

	while (true) {
		new_nodes_list_lock_.lock();

		int i = new_nodes_list_shared_idx_;
//		if (i == (int)new_nodes_.size()) {
		if (i == (int)new_nodes_size_) {
			new_nodes_list_lock_.unlock();
			if (helper_threads_mode_ == 1) {
				std::this_thread::yield();
				std::this_thread::sleep_for(std::chrono::microseconds(20));
				continue;
			} else {
				break;
			}
		}
		int n = new_nodes_size_ - i;
		if (n > 5) n = 5;
		new_nodes_list_shared_idx_ += n;
		new_nodes_list_lock_.unlock();

		for (; n > 0; n--, i++) {

		Node_revamp* newchild = new_nodes_[i].node;
		//int idx = new_nodes_[i].idx;

		count++;

		int nappends = appendHistoryFromTo(movestack, history, root_node_, newchild);
		//Node_revamp* newchild = node->GetEdges()[idx].GetChild();

		//history->Append(node->GetEdges()[idx].move_);

		newchild->ExtendNode(history, root_node_);

		if (!newchild->IsTerminal()) {

			//AddNodeToComputation(&history);
			auto planes = EncodePositionForNN(*history, 8, history_fill_);

			computation_lock_.lock();
			computation_->AddInput(std::move(planes));
			new_nodes_[i].batch_idx = minibatch_shared_idx_++;
			//minibatch_.push_back({newchild, (uint16_t)i});
			//LOGFILE << "minibatch add: " << new_nodes_[i].junction;
			computation_lock_.unlock();

		} else {  // is terminal
			//non_computation_lock_.lock();
			//non_computation_new_nodes_.push_back({newchild, (uint16_t)i});
			//non_computation_lock_.unlock();
		}

		history->Trim(played_history_length_);
		//for (int j = 0; j <= nappends; j++) {
		//	history->Pop();
		//}

		// not checking and setting N = 0 (see code that propagates below) here means duplicates can exist in the queue if MULTIPLE_NEW_SIBLINGS = true
		// but checking for duplicates that way does not work with multiple threads because N values are not restored until after the nn-computation (and meanwhile other threads can run)

		if (nappends - 1 > full_tree_depth) full_tree_depth = nappends - 1;
		cum_depth += nappends - 1;

		}
	}

//	search_->counters_lock_.lock();
	if (full_tree_depth > search_->full_tree_depth_) search_->full_tree_depth_ = full_tree_depth;
	search_->cum_depth_ += cum_depth;
//	search_->counters_lock_.unlock();

	return count;
}


void SearchWorker_revamp::retrieveNNResult(Node_revamp* node, int batchidx) {
  float q = -computation_->GetQVal(batchidx);
  if (q < -1.0 || q > 1.0) {
    std::cerr << "q = " << q << "\n";
    abort();
    //if (q < -1.0) q = -1.0;
    //if (q > 1.0) q = 1.0;
  }
  node->SetOrigQ(q);

  float total = 0.0;
  int nedge = node->GetNumEdges();
  pvals_.clear();
  for (int k = 0; k < nedge; k++) {
    float p = computation_->GetPVal(batchidx, (node->GetEdges())[k].move_.as_nn_index());
    if (p < 0.0) {
      std::cerr << "p value < 0\n";
      abort();
      //p = 0.0;
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
  for (int i = 0; i < node->GetNumChildren(); i++) {
    q -= node->GetEdges()[i].GetChild()->GetW() * node->GetEdges()[i].GetChild()->GetQ();
  }
  node->SetQ(q);
  // Average Q STOP

  
  int n = 1;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    n += node->GetEdges()[i].GetChild()->GetN();
  }
  node->SetN(n);

	int first_non_created_child_idx = node->GetNumChildren();
	while (first_non_created_child_idx < node->GetNumEdges() && node->GetEdges()[first_non_created_child_idx].GetChild() != nullptr) {
		first_non_created_child_idx++;
	}

//  if (MULTIPLE_NEW_SIBLINGS)
//    n = node->GetNumEdges() - first_non_created_child_idx;
//  else
//    n = node->GetNumEdges() > first_non_created_child_idx ? 1 : 0;

//  for (int i = 0; i < node->GetNumChildren(); i++) {
//    n += node->GetEdges()[i].GetChild()->GetNExtendable();
//  }
//  node->SetNExtendable(n);

	int16_t max_idx = -1;
	float max_w = 0.0;
	int nidx = node->GetNextUnexpandedEdge();
	if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
		max_w = node->GetEdges()[nidx].GetP();
	}
	for (int i = 0; i < node->GetNumChildren(); i++) {
		float br_max_w = node->GetEdges()[i].GetChild()->GetW() * node->GetEdges()[i].GetChild()->GetMaxW();
		if (br_max_w > max_w) {
			max_w = br_max_w;
			max_idx = i;
		}
	}
	node->SetMaxW(max_w);
	node->SetBestIdx(max_idx);
}


int SearchWorker_revamp::propagate() {
	int count = 0;

	//auto start_comp_time = std::chrono::steady_clock::now();
	//auto stop_comp_time = std::chrono::steady_clock::now();

	while (true) {
		new_nodes_list_lock_.lock();
		int j = new_nodes_list_shared_idx_;
		if (j == new_nodes_amount_retrieved_) {
			new_nodes_list_lock_.unlock();
			if (helper_threads_mode_ == 3) {
				std::this_thread::yield();
				std::this_thread::sleep_for(std::chrono::microseconds(20));
				continue;
			} else {
				break;
			}
		}
		int n = new_nodes_amount_retrieved_ - j;
		if (n > 5) n = 5;
		new_nodes_list_shared_idx_ += n;
		new_nodes_list_lock_.unlock();

		for (; n > 0; n--, j++) {
			Node_revamp* node = new_nodes_[j].node->GetParent();
			uint16_t juncidx = new_nodes_[j].junction;

			//LOGFILE << "node: " << node << ", juncidx: " << juncidx;

			while (juncidx != 0xFFFF) {
				while (node != junctions_[juncidx].node) {
					recalcPropagatedQ(node);
					count++;
					node = node->GetParent();
				}
				junction_locks_[juncidx]->lock();
				int children_count = --junctions_[juncidx].children_count;
				junction_locks_[juncidx]->unlock();
				if (children_count > 0) break;
				juncidx = junctions_[juncidx].parent;
			}
			if (juncidx == 0xFFFF) {
				while (true) {
					recalcPropagatedQ(node);
					count++;
					if (node == root_node_) break;
					node = node->GetParent();
				}
			}
		}
	}

		//~ while (true) {
			//~ node = node->GetParent();

			//~ uint16_t &br = branching_[node];

			//~ start_comp_time = std::chrono::steady_clock::now();
			//~ branching_lock_.lock();
			//~ stop_comp_time = std::chrono::steady_clock::now();
			//~ duration_node_prio_queue_lock_ += (stop_comp_time - start_comp_time).count();

			//~ int b = --br;

			//~ branching_lock_.unlock();

			//~ if (b > 0) break;
			//~ recalcPropagatedQ(node);
			//~ count++;
			//~ if (node == root_node_) break;
		//~ }
	//}
	return count;
}



void SearchWorker_revamp::ThreadLoop(int thread_id) {

	PositionHistory history(search_->played_history_);
	std::vector<Move> movestack;

	new_nodes_ = new NewNode[batch_size_];

	search_->busy_mutex_.lock();
	if (LOG_RUNNING_INFO) LOGFILE << "Working thread: " << thread_id;

	std::vector<std::mutex *> helper_thread_locks;
	std::vector<std::thread> helper_threads;
	for (int j = 0; j < N_HELPER_THREADS; j++) {
		helper_thread_locks.push_back(new std::mutex());
		helper_thread_locks[j]->lock();
		std::mutex *lock = helper_thread_locks[j];
    helper_threads.emplace_back([this, j, lock]()
      {
        HelperThreadLoop(j, lock);
      }
    );
  }

	for (int n = batch_size_; n > 0; n--) {
		junction_locks_.push_back(new std::mutex());
	}

//  auto board = history.Last().GetBoard();
//  if (DEBUG) LOGFILE << "Inital board:\n" << board.DebugString();

//  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

//  unsigned int lim = limits_.visits;

//  int i = 0;

  if (root_node_->GetNumEdges() == 0 && !root_node_->IsTerminal()) {  // root node not extended
    root_node_->ExtendNode(&history, root_node_);
    if (root_node_->IsTerminal()) {
      std::cerr << "Root " << root_node_ << " is terminal, nothing to do\n";
      abort();
    }
    computation_ = search_->network_->NewComputation();
    AddNodeToComputation(&history);

    // LOGFILE << "Computing thread root ..";
    computation_->ComputeBlocking();
    // LOGFILE << " done\n";
    retrieveNNResult(root_node_, 0);
    //i++;
  }

//	auto cmp = [](PropagateQueueElement left, PropagateQueueElement right) { return left.depth < right.depth;};
//	std::priority_queue<PropagateQueueElement, std::vector<PropagateQueueElement>, decltype(cmp)> propagate_queue(cmp);


	while (search_->not_stop_searching_) {

		computation_ = search_->network_->NewComputation();

		helper_threads_mode_ = 1;
		//LOGFILE << "Allowing helper threads to help";
		for (int j = 0; j < (int)helper_thread_locks.size(); j++) {
			helper_thread_locks[j]->unlock();
		}

		auto start_comp_time = std::chrono::steady_clock::now();
		//auto start_comp_time2 = start_comp_time;

		//LOGFILE << "Working myself.";
    pickNodesToExtend();

		helper_threads_mode_ = 2;  // from now no new nodes will be added

//		if (new_nodes_.size() == 0) {  // no new nodes found, but there may exist unextended edges unavailable due to business
		if (new_nodes_size_ == 0) {  // no new nodes found, but there may exist unextended edges unavailable due to business
			for (int j = 0; j < (int)helper_thread_locks.size(); j++) {
				helper_thread_locks[j]->lock();
			}
			if (search_->half_done_count_ == 0) {  // no other thread is waiting for nn computation and new nodes to finish so the search tree is exhausted
				search_->not_stop_searching_ = false;
				break;
			}

			search_->busy_mutex_.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			search_->busy_mutex_.lock();
			continue;
		}

		buildJunctionRTree();

		junction_of_node_.clear();

		auto stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_search_ += (stop_comp_time - start_comp_time).count();



		start_comp_time = std::chrono::steady_clock::now();

		int count = extendTree(&movestack, &history);

		if (LOG_RUNNING_INFO) LOGFILE << "main thread new nodes: " << count;

		for (int j = 0; j < (int)helper_thread_locks.size(); j++) {
			helper_thread_locks[j]->lock();
		}

		//if (non_computation_new_nodes_.size() > 0) {
		//	LOGFILE << "terminal node!!";
		//}
		//for (int i = (int)non_computation_new_nodes_.size() - 1; i >= 0; i--) {
		//	uint32_t juncidx = new_nodes_[non_computation_new_nodes_[i].new_nodes_idx].junction;
		//	while (juncidx != 0xFFFF) {
		//		junctions_[juncidx].children_count--;
		//		if (junctions_[juncidx].children_count > 0) break;
		//		juncidx = junctions_[juncidx].parent;
		//	}
		//	Node_revamp* node = non_computation_new_nodes_[i].node;
		//	while (true) {
		//		node = node->GetParent();
		//		recalcPropagatedQ(node);
		//		if (node == root_node_) break;
		//	}
		//}
		//non_computation_new_nodes_.clear();

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_create_ += (stop_comp_time - start_comp_time).count();

		//~ if (minibatch_.size() < propagate_list_.size()) {
			//~ std::cerr << "minibatch_.size() < propagate_list_.size(): " << minibatch_.size() << " < " << propagate_list_.size() << "\n";
			//~ abort();
		//~ }

		new_nodes_list_shared_idx_ = 0;

		if (LOG_RUNNING_INFO) LOGFILE
						<< "n: " << root_node_->GetN()
//						<< ", new_nodes_ size: " << new_nodes_.size()
						<< ", new_nodes_ size: " << new_nodes_size_
						//<< ", minibatch_ size: " << minibatch_.size()
						<< ", junctions_ size: " << junctions_.size();
						//<< ", highest w: " << new_nodes_[new_nodes_.size() - 1].w
						//<< ", node stack size: " << nodestack_.size()
						//<< ", max_unexpanded_w: " << new_nodes_[0];

		int my_iteration = search_->iteration_count_a_++;
		search_->half_done_count_ += new_nodes_size_;

		//LOGFILE << "Unlock " << thread_id;
		search_->busy_mutex_.unlock();

    // std::this_thread::sleep_for(std::chrono::milliseconds(0));
		start_comp_time = std::chrono::steady_clock::now();

		if (minibatch_shared_idx_ > 0) {
			computation_->ComputeBlocking();
			minibatch_shared_idx_ = 0;
		}

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_compute_ += (stop_comp_time - start_comp_time).count();

		search_->busy_mutex_.lock();

		while (search_->iteration_count_b_ != my_iteration) {
			search_->busy_mutex_.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			search_->busy_mutex_.lock();
		}
		search_->iteration_count_b_++;
		search_->half_done_count_ -= new_nodes_size_;

		if (LOG_RUNNING_INFO) LOGFILE << "Working thread: " << thread_id;


		//i += minibatch.size();

		start_comp_time = std::chrono::steady_clock::now();

		helper_threads_mode_ = 3;
		for (int j = 0; j < (int)helper_thread_locks.size(); j++) {
			helper_thread_locks[j]->unlock();
		}

		for (int j = 0; j < (int)new_nodes_size_; j++) {
			new_nodes_[j].node->IncrParentNumChildren();
			if (new_nodes_[j].batch_idx != -1) {
				retrieveNNResult(new_nodes_[j].node, new_nodes_[j].batch_idx);
			}
			new_nodes_amount_retrieved_++;
		}

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_retrieve_ += (stop_comp_time - start_comp_time).count();

		helper_threads_mode_ = 4;

		start_comp_time = std::chrono::steady_clock::now();

		int pcount = propagate();

		for (int j = 0; j < (int)helper_thread_locks.size(); j++) {
			helper_thread_locks[j]->lock();
		}

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_propagate_ += (stop_comp_time - start_comp_time).count();
		search_->count_iterations_++;

		if (LOG_RUNNING_INFO) LOGFILE << "main thread did propagates: " << pcount;

		new_nodes_list_shared_idx_ = 0;
		new_nodes_amount_retrieved_ = 0;

		junctions_.clear();

		//new_nodes_.clear();
		new_nodes_size_ = 0;

		int64_t time = search_->GetTimeSinceStart();
		if (time - search_->last_uci_time_ > kUciInfoMinimumFrequencyMs) {
			search_->last_uci_time_ = time;
			search_->SendUciInfo();
		}

		if (search_->not_stop_searching_) {
			search_->checkLimitsAndMaybeTriggerStop();
		}
  }

	//search_->threads_list_mutex_.lock();
	int nt = --search_->n_thread_active_;
	//search_->threads_list_mutex_.unlock();

	if (nt == 0) {  // this is the last thread
		int64_t elapsed_time = search_->GetTimeSinceStart();
		//LOGFILE << "Elapsed time when thread for node " << root_node_ << " which has size " << root_node_->GetN() << " nodes did " << i << " computations: " << elapsed_time << "ms";
		LOGFILE << "Elapsed time for " << root_node_->GetN() << " nodes: " << elapsed_time << "ms";
		LOGFILE << "#helper threads: " << N_HELPER_THREADS;

		LOGFILE << "root Q: " << root_node_->GetQ();

//		LOGFILE << "move   P                 n   norm n      h   Q          w";
		LOGFILE << "move   P                 n   norm n     Q          w";
		for (int i = 0; i < root_node_->GetNumChildren(); i++) {
			LOGFILE << std::fixed << std::setfill(' ') 
								<< (root_node_->GetEdges())[i].move_.as_string() << " "
								<< std::setw(10) << (root_node_->GetEdges())[i].GetP() << " "
								<< std::setw(10) << (root_node_->GetEdges())[i].GetChild()->GetN() << " "
								<< std::setw(10) << (float)(root_node_->GetEdges())[i].GetChild()->GetN() / (float)(root_node_->GetN() - 1) << " "
//								<< std::setw(4) << (root_node_->GetEdges())[i].GetChild()->ComputeHeight() << " "
								<< std::setw(10) << (float)(root_node_->GetEdges())[i].GetChild()->GetQ() << " "
								<< std::setw(10) << root_node_->GetEdges()[i].GetChild()->GetW();
		}

		if (search_->count_iterations_ > 0) {
			LOGFILE << "search: " << search_->duration_search_ / search_->count_iterations_
							<< ", create: " << search_->duration_create_ / search_->count_iterations_
							<< ", compute: " << search_->duration_compute_ / search_->count_iterations_
							<< ", retrieve: " << search_->duration_retrieve_ / search_->count_iterations_
							<< ", propagate: " << search_->duration_propagate_ / search_->count_iterations_;
							//<< ", duration_node_prio_queue_lock_: " << duration_node_prio_queue_lock_ / count_iterations_;
		}

		int64_t dur_sum = (search_->duration_search_ + search_->duration_create_ + search_->duration_compute_ + search_->duration_retrieve_ + search_->duration_propagate_) / 1000;

		if(dur_sum > 0){ // I've seen cases with dur_sum == 0
		  LOGFILE << "search: " << search_->duration_search_ / dur_sum
						<< ", create: " << search_->duration_create_ / dur_sum
						<< ", compute: " << search_->duration_compute_ / dur_sum
						<< ", retrieve: " << search_->duration_retrieve_ / dur_sum
						<< ", propagate: " << search_->duration_propagate_ / dur_sum;
		}

		if (!search_->ponder_ && !search_->abort_) {
			int bestidx = indexOfHighestQEdge(root_node_);
			Move best_move = root_node_->GetEdges()[bestidx].GetMove(search_->played_history_.IsBlackToMove());
			int ponderidx = indexOfHighestQEdge(root_node_->GetEdges()[bestidx].GetChild());
			// If the move we make is terminal, then there is nothing to ponder about
			if(!root_node_->GetEdges()[bestidx].GetChild()->IsTerminal()){
				Move ponder_move = root_node_->GetEdges()[bestidx].GetChild()->GetEdges()[ponderidx].GetMove(!search_->played_history_.IsBlackToMove());
				search_->best_move_callback_({best_move, ponder_move});    
			} else {
				search_->best_move_callback_({best_move});    
			}
		}
	}

	while (!junction_locks_.empty()) {
		delete junction_locks_.back();
		junction_locks_.pop_back();
	}

	helper_threads_mode_ = -1;
  while (!helper_threads.empty()) {
		helper_thread_locks.back()->unlock();
		helper_threads.back().join();
		delete helper_thread_locks.back();
		helper_thread_locks.pop_back();
		helper_threads.pop_back();
  }

	delete[] new_nodes_;

	//LOGFILE << "Unlock " << thread_id;
	search_->busy_mutex_.unlock();
}


void SearchWorker_revamp::HelperThreadLoop(int helper_thread_id, std::mutex* lock) {
	PositionHistory history(search_->played_history_);
	std::vector<Move> movestack;

	while (true) {
		lock->lock();

		if (helper_threads_mode_ == 1 || helper_threads_mode_ == 2) {
			int count = extendTree(&movestack, &history);
			if (LOG_RUNNING_INFO) if (count > 0) LOGFILE << "helper thread " << helper_thread_id << " did new nodes: " << count;
		} else {
			if (helper_threads_mode_ == 3 || helper_threads_mode_ == 4) {
				int count = propagate();
				if (LOG_RUNNING_INFO) LOGFILE << "helper thread " << helper_thread_id << " did propagates: " << count;
			} else
				if (helper_threads_mode_ == -1) {
					lock->unlock();
					break;
				} else {
					std::cerr << helper_threads_mode_ << " kjqekje\n";
					abort();
				}
		}

		lock->unlock();
		std::this_thread::sleep_for(std::chrono::microseconds(20));
		std::this_thread::yield();
	}
}



}  // namespace lczero
