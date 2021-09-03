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

#include "glow/search.h"
#include "glow/search_worker.h"
#include "glow/strategy.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <cassert>  // assert() used for debugging during development
#include <iomanip>
#include <random>

#include "neural/encoder.h"

namespace lczero {

namespace {

    std::random_device r;
    std::default_random_engine eng{r()};
    std::uniform_real_distribution<double> urd(0, 1);
  

// Alternatives:

  // int const MAX_NEW_SIBLINGS = 1000;
  int const MAX_NEW_SIBLINGS = 1;
  // The maximum number of new siblings. If 1, then it's like old MULTIPLE_NEW_SIBLINGS = false, if >= maximum_number_of_legal_moves it's like MULTIPLE_NEW_SIBLINGS = true
const int kUciInfoMinimumFrequencyMs = 5000;

int const N_HELPER_THREADS_PRE = 3;
int const N_HELPER_THREADS_POST = 3;

bool const LOG_RUNNING_INFO = false;

  // with this set to true pickNodesToExtend() is not used
  // bool const OLD_PICK_N_CREATE_MODE = false;
  bool const OLD_PICK_N_CREATE_MODE = true;

}  // namespace


  NodeGlow* SearchWorkerGlow::GetInterestingChild(NodeGlow* node) {
    // pick an interesting child based on Weight and Policy.

    int num_children = node->GetNumChildren();
    std::vector<double> effective_weights(num_children, 0.0f);
    double sum_of_effective_weights = 0;
    double sum_of_policy_of_extended_nodes = 0;

    // if there are less than two edges extended, return fast
    if(num_children == 0){
      return(nullptr);
    }

    if(node->GetNumEdges() == 1){
      // Only one legal move
      if(num_children == 1){
	return(node->GetFirstChild());
      } else {
	return(nullptr);
      }
    }

    // Sum the policy of extended children
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      float policy = node->GetEdges()[i->GetIndex()].GetP();
      sum_of_policy_of_extended_nodes += policy;
    }

    double the_extend_sample = urd(eng);

    // If there are unextended children, then first compare the_sample to sum_of_policy_of_extended_nodes, and if the_sample is higher, then return nullptr.
    if(num_children < node->GetNumEdges()){
      if(the_extend_sample > sum_of_policy_of_extended_nodes){
	return(nullptr);
      }
    }

    // ignore the unextended children, the winner must be an extended child.
    // Calculate weights for extended children
    
    // Let policy influence which children are traversed and which leaves are extended, but make sure there is room for the q-signal to change the final distribution, or else we would
    // get into self-reinforcement loops where policy drifts unboundedly to sharper and sharper distributions. Since we use approx 800 nodes per move in training, it sounds reasonable
    // to let policy affect node selection only until one fourth of that budget is used. Note, however, that this is at root, in tree most nodes will have an substantial policy influence even
    // after that. To make sure the q-signal will help the network to learn, cap the policy influence to 0.5 regardless of number of visits to the parent.
    float n = 0.0f;
    float policy_weight_starting_point = 0.5; // Let policy weigh this much when visits is 1.
    float policy_decay = 200.0; // after this number of visits, forget about policy.
    float policy_weight = std::max(n, (policy_decay - node->GetN()))/policy_decay * policy_weight_starting_point;
    float weight_weight = 1 - policy_weight;
    // if(node->GetN() % 50 == 0){
      // LOGFILE << "policy_weight: " << policy_weight << " visits: " << node->GetN();
    // }
    float highest_weight_weight = 0.0f;
    int index_of_node_with_highest_weight_weight = 0;
    int j = 0;
    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      effective_weights[i->GetIndex()] = i->GetW() * weight_weight + policy_weight * node->GetEdges()[i->GetIndex()].GetP();
      sum_of_effective_weights += effective_weights[i->GetIndex()];
      if(i->GetW() >= highest_weight_weight){
	index_of_node_with_highest_weight_weight = j;
	index_of_node_with_highest_weight_weight = i->GetW();
      }
      j += 1;
      // LOGFILE << "at child " << i->GetIndex() << " with policy " << node->GetEdges()[i->GetIndex()].GetP() << " and weight " << i->GetW() << " and visits " << i->GetN() << " effective weight " << effective_weights[i->GetIndex()];
    }
    LOGFILE << "Q of best child: " << highest_weight_weight;

    // exploitation or exploration?
    double exploitation_sample = urd(eng);
    if(exploitation_sample > 0.5){
      // Just go with best child
      j = 0;
      for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
	// make sure one child is choosen, even if there are numerical problems (the sample is 1 and the sum of effective weights never quite reaches one.
	if(j == index_of_node_with_highest_weight_weight){
	  LOGFILE << "Greedy: returning node" << j << " with Q: " << i->GetW();
	  return(i);
	}
	j += 1;
      }
    }
    
    double scaler = 1/sum_of_effective_weights;
    
    // scale weights so they sum to 1.
    std::transform(effective_weights.begin(), effective_weights.end(), effective_weights.begin(), [&scaler](auto& c){return c*scaler;});

    sum_of_effective_weights = 0;

    double the_select_child_sample = urd(eng);

    for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
      sum_of_effective_weights += effective_weights[i->GetIndex()];
      // make sure one child is choosen, even if there are numerical problems (the sample is 1 and the sum of effective weights never quite reaches one.
      if((sum_of_effective_weights >= the_select_child_sample) || (i->GetNextSibling() == nullptr)){
	LOGFILE << "Not Greedy: returning node with Q: " << i->GetW();
	return(i);
      }
    }
    
    LOGFILE << "No interesting child found!";
    abort();
  }

void SearchWorkerGlow::pickNodesToExtend() {
	NodeGlow* node;
	NodeGlow *best_child;

	int nodes_visited = 0;
	int depth = 0;

	for (int n = 0; n < new_nodes_amount_target_ && n < new_nodes_amount_limit_; n++) {
		node = root_node_;

		while (true) {
			nodes_visited++;
			// best_child = node->GetBestChild();
			best_child = GetInterestingChild(node);
			
			if (best_child == nullptr) {
				int nidx = node->GetNextUnexpandedEdge();
				if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
					new_nodes_[new_nodes_size_] = {std::make_unique<NodeGlow>(node, nidx), node, 0xFFFF, -1};
					new_nodes_size_++;
					node->SetNextUnexpandedEdge(nidx + 1);
					break;
				} else {  // no more child to add (before retrieved information about previous ones)
					return;
				}
			}
			node = best_child;
			depth++;
		}

		int junction_mode = 0;
		uint16_t ccidx = (new_nodes_size_ - 1) | 0x8000;

		while (true) {
			recalcMaxW(node);

			if (junction_mode == 0) {
				uint16_t n = node->GetBranchingInFlight();
				if (n == 0) {  // an unvisited node
					node->SetBranchingInFlight(ccidx);
				} else if (n & 0xC000) {  // part of a path between junctions
					uint16_t new_junc_idx = junctions_.size();
					node->SetBranchingInFlight(new_junc_idx + 1);
					uint16_t parent;
					if (n & 0x8000) {
						parent = new_nodes_[n & 0x3FFF].junction;
						new_nodes_[n & 0x3FFF].junction = new_junc_idx;
					} else {
						parent = junctions_[n & 0x3FFF].parent;
						junctions_[n & 0x3FFF].parent = new_junc_idx;
					}
					junctions_.push_back({node, parent, 0});

					if (ccidx & 0x8000) {
						new_nodes_[ccidx & 0x3FFF].junction = new_junc_idx;
					} else {
						junctions_[ccidx & 0x3FFF].parent = new_junc_idx;
					}
					ccidx = new_junc_idx | 0x4000;
					junction_mode = 1;
                                } else {  // a junction node
					if (ccidx & 0x8000) {
						new_nodes_[ccidx & 0x3FFF].junction = n - 1;
					} else {
						junctions_[ccidx & 0x3FFF].parent = n - 1;
					}
					junction_mode = 2;
				}
			} else if (junction_mode == 1) {
				uint16_t n = node->GetBranchingInFlight();
				if (n & 0xC000) {  // part of path between junctions
					node->SetBranchingInFlight(ccidx);
				} else {  // a junction node
					junction_mode = 2;
				}
			}

			if (node == root_node_) break;
			node = node->GetParent();
		}
	}

  search_->count_search_node_visits_ += nodes_visited;
}

void SearchWorkerGlow::buildJunctionRTree() {
	for (int i = new_nodes_size_ - 1; i >= 0; i--) {
		uint16_t junction = new_nodes_[i].junction;
		while (junction != 0xFFFF) {
			int cc = junctions_[junction].children_count++;
			if (cc > 0) break;
			junction = junctions_[junction].parent;
		}
	}

//	for (int i = new_nodes_.size() - 1; i >= 0; i--) {
	for (int i = new_nodes_size_ - 1; i >= 0; i--) {
		NodeGlow* node = new_nodes_[i].parent;
		while (node->GetBranchingInFlight() != 0) {
			node->SetBranchingInFlight(0);
			if (node == root_node_) break;
			node = node->GetParent();
		}
	}
}

int SearchWorkerGlow::appendHistoryFromTo(std::vector<Move> *movestack, PositionHistory *history, NodeGlow* from, NodeGlow* to) {
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


int SearchWorkerGlow::AddNodeToComputation(NodeGlow* node, PositionHistory *history) {
  auto hash = history->HashLast(cache_history_length_plus_1_);
  auto planes = EncodePositionForNN(*history, 8, history_fill_);
  int nedge = node->GetNumEdges();
  std::vector<uint16_t> moves;
  moves.reserve(nedge);
  for (int k = 0; k < nedge; k++) {
    moves.emplace_back(node->GetEdges()[k].move_.as_nn_index());
  }
	computation_lock_.lock();
  if (computation_->AddInputByHash(hash)) {
    new_nodes_amount_target_++;  // it's cached so it shouldn't be counted towards the minibatch size
  } else {
    //computation_->AddInput(std::move(planes));
    computation_->AddInput(hash, std::move(planes), std::move(moves));
  }
  int idx = minibatch_shared_idx_++;
	computation_lock_.unlock();
	return idx;
}

int SearchWorkerGlow::MaybeAddNodeToComputation(NodeGlow* node, PositionHistory *history) {
  auto hash = history->HashLast(cache_history_length_plus_1_);
  {
    NNCacheLock nneval(search_->cache_, hash);
    if (nneval) {  // it's cached
      float q = -nneval->q;
      node->SetOrigQ(q);
      int nedge = node->GetNumEdges();
      for (int k = 0; k < nedge; k++) {
	node->GetEdges()[k].SetP(nneval->p[k].second);
      }
      
      node->SetMaxW(node->GetEdges()[0].GetP()); // Why would Policy of the first node be the weight?
      
      return -1;
    }
  }
  auto planes = EncodePositionForNN(*history, 8, history_fill_);
  int nedge = node->GetNumEdges();
  std::vector<uint16_t> moves;
  moves.reserve(nedge);
  for (int k = 0; k < nedge; k++) {
    moves.emplace_back(node->GetEdges()[k].move_.as_nn_index());
  }  
  computation_lock_.lock();
  //computation_->AddInput(std::move(planes));
  computation_->AddInput(hash, std::move(planes), std::move(moves));
  int idx = minibatch_shared_idx_++;
  computation_lock_.unlock();
  return idx;
}


int SearchWorkerGlow::extendTree(std::vector<Move> *movestack, PositionHistory *history) {
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
		if (n > 10) n = 10;
		new_nodes_list_shared_idx_ += n;
		new_nodes_list_lock_.unlock();

		for (; n > 0; n--, i++) {

			NodeGlow* newchild = new_nodes_[i].new_node.get();
			//int idx = new_nodes_[i].idx;

			count++;

			int nappends = appendHistoryFromTo(movestack, history, root_node_, newchild);
			//NodeGlow* newchild = node->GetEdges()[idx].GetChild();

			//history->Append(node->GetEdges()[idx].move_);

			search_->ExtendNode(history, newchild);

			if (!newchild->IsTerminal()) {

// 				int idx = AddNodeToComputation(newchild, history);
// 				new_nodes_[i].batch_idx = idx;

				int idx = MaybeAddNodeToComputation(newchild, history);
				if (idx == -1) {
					new_nodes_amount_target_++;  // it's cached so it shouldn't be counted towards the minibatch size
				} else {
					new_nodes_[i].batch_idx = idx;
				}
			} else {  // is terminal
				new_nodes_amount_target_++;  // it's terminal so it shouldn't be counted towards the minibatch size
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


void SearchWorkerGlow::picknextend(PositionHistory *history) {

//	std::vector<int> path;
//	unsigned int last_depth = 0;
	int nodes_visited = 0;
//	bool same_path = false;
	int full_tree_depth = search_->full_tree_depth_;
	int cum_depth = 0;

	// turn on global tree lock
	while (new_nodes_size_ < new_nodes_amount_target_ && new_nodes_size_ < new_nodes_amount_limit_) {  // repeat this until minibatch_size amount of non terminal, non cache hit nodes have been found (or reached a predefined limit larger than minibatch size)
		NodeGlow *node = root_node_;
		NodeGlow *best_child = node->GetBestChild();
		// NodeGlow *best_child = GetInterestingChild(node, cum_depth);
		if (best_child == nullptr && (node->GetNextUnexpandedEdge() == node->GetNumEdges() || node->GetNextUnexpandedEdge() - node->GetNumChildren() == MAX_NEW_SIBLINGS)) break;  // no more expandable node
		// starting from root node follow maxidx until next move would make the sub tree to small
		// propagate no availability upwards to root
		// turn off global tree lock
		//for (;;) {  // repeat until localbatch_size amount of nodes have been found
			// go down to max unexpanded node
//			unsigned int depth = 0;
			while (true) {
				if (best_child == nullptr) break;  // best unexpanded node is child of this node
// 				if (same_path) {
// 					if (depth == last_depth) {  // reached end of last path without deviating
// 						same_path = false;
// 						if (path.size() == depth) {
// 							path.push_back(best_idx);
// 						} else {
// 							path[depth] = best_idx;
// 						}
// 						history->Pop();  // pop move of last new node
// 						history->Append(node->GetEdges()[best_idx].move_);
// 						nodes_visited++;
// 					} else {
// 						if (best_idx != path[depth]) {  // deviates from last path
// 							same_path = false;
// 							path[depth] = best_idx;
// 							history->Trim(played_history_length_ + depth);
// 							history->Append(node->GetEdges()[best_idx].move_);
// 							nodes_visited++;
// 						}
// 					}
// 				} else {  // not same path
// 					if (path.size() == depth) {
// 						path.push_back(best_idx);
// 					} else {
// 						path[depth] = best_idx;
// 					}
					history->Append(node->GetEdges()[best_child->GetIndex()].move_);
					nodes_visited++;
// 				}
				node = best_child;
				best_child = node->GetBestChild();
//				depth++;
			};

			nodes_visited++;
			
// 			if (same_path) {
// 				history->Trim(played_history_length_ + depth);
// 			}
			
// 			same_path = true;
// 			last_depth = depth;
			
			int nidx = node->GetNextUnexpandedEdge();
			history->Append(node->GetEdges()[nidx].move_);
			node->SetNextUnexpandedEdge(nidx + 1);

			std::unique_ptr<NodeGlow>newnode = std::make_unique<NodeGlow>(node, nidx);

			bool out_of_order = false;
			int nnidx = -1;

			search_->ExtendNode(history, newnode.get());
			if (newnode.get()->IsTerminal()) {
				out_of_order = true;
				node->AddChild(std::move(newnode));
			} else {
// 				int16_t batchidx = AddNodeToComputation(newnode.get(), history);
// 				nnidx = new_nodes_size_++;
// 				new_nodes_[nnidx] = {std::move(newnode), node, 0xFFFF, batchidx};

				int16_t batchidx = MaybeAddNodeToComputation(newnode.get(), history);
				if (batchidx == -1) {
					out_of_order = true;
					node->AddChild(std::move(newnode));
				} else {
					nnidx = new_nodes_size_++;
					new_nodes_[nnidx] = {std::move(newnode), node, 0xFFFF, batchidx};
				}
			}
			int depth = history->GetLength() - played_history_length_;
			history->Trim(played_history_length_);

			if (depth > full_tree_depth) full_tree_depth = depth;
			cum_depth += depth;

			if (out_of_order) {
				while (true) {
					recalcPropagatedQ(node);
					if (node == root_node_) break;
					node = node->GetParent();
				}
			} else {  // not out of order
				int junction_mode = 0;
				uint16_t ccidx = nnidx | 0x8000;

				while (true) {
					recalcMaxW(node);

					if (junction_mode == 0) {
						uint16_t n = node->GetBranchingInFlight();
						if (n == 0) {  // an unvisited node
							node->SetBranchingInFlight(ccidx);
						} else if (n & 0xC000) {  // part of a path between junctions
							uint16_t new_junc_idx = junctions_.size();
							node->SetBranchingInFlight(new_junc_idx + 1);
							uint16_t parent;
							if (n & 0x8000) {
								parent = new_nodes_[n & 0x3FFF].junction;
								new_nodes_[n & 0x3FFF].junction = new_junc_idx;
							} else {
								parent = junctions_[n & 0x3FFF].parent;
								junctions_[n & 0x3FFF].parent = new_junc_idx;
							}
							junctions_.push_back({node, parent, 0});

							if (ccidx & 0x8000) {
								new_nodes_[ccidx & 0x3FFF].junction = new_junc_idx;
							} else {
								junctions_[ccidx & 0x3FFF].parent = new_junc_idx;
							}
							ccidx = new_junc_idx | 0x4000;
							junction_mode = 1;
						} else {  // a junction node
							if (ccidx & 0x8000) {
								new_nodes_[ccidx & 0x3FFF].junction = n - 1;
							} else {
								junctions_[ccidx & 0x3FFF].parent = n - 1;
							}
							junction_mode = 2;
						}
					} else if (junction_mode == 1) {
						uint16_t n = node->GetBranchingInFlight();
						if (n & 0xC000) {  // part of path between junctions
							node->SetBranchingInFlight(ccidx);
						} else {  // a junction node
							junction_mode = 2;
						}
					}

					if (node == root_node_) break;
					node = node->GetParent();
				}
			}

			// when deviating from previous path, trim history according to this and start pushing each new move
			// create node, increment inner loop node count
			// compute legal moves
			// if node is terminal or cache hit, set its q (and p:s) and turn on full propagation mode
			// otherwise, add node to computation list and turn on limited max only propagation and forking tree mode. Increment computation node count.
			// propagate to local tree root according to mode
		//}
		// turn on global tree lock
		// propagate to root
	}

// 	history->Trim(played_history_length_);

	if (full_tree_depth > search_->full_tree_depth_) search_->full_tree_depth_ = full_tree_depth;
	search_->cum_depth_ += cum_depth;

	search_->count_search_node_visits_ += nodes_visited;

}

void SearchWorkerGlow::retrieveNNResult(NodeGlow* node, int batchidx) {
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
    assert((p >= 0.0) && (p <= 1.0));
    if (p_concentration_ != 1.0f) {
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


inline void SearchWorkerGlow::recalcMaxW(NodeGlow *node) {
	NodeGlow *max_child = nullptr;
	float max_w = 0.0;
	int nidx = node->GetNextUnexpandedEdge();
	if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
		max_w = node->GetEdges()[nidx].GetP();
	}
	for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
		float br_max_w = i->GetW() * i->GetMaxW();
		if (br_max_w > max_w) {
			max_w = br_max_w;
			max_child = i;
		}
	}
	node->SetMaxW(max_w);
	node->SetBestChild(max_child);
}

void SearchWorkerGlow::recalcPropagatedQ(NodeGlow* node) {
  int n = 1;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    n += i->GetN();
  }
  node->SetN(n);

  float q = compute_q_and_weights(node);
	node->SetQ(q);

	recalcMaxW(node);
}


int SearchWorkerGlow::propagate() {
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
		if (n > 1) n = 1;
		new_nodes_list_shared_idx_ += n;
		new_nodes_list_lock_.unlock();

		for (; n > 0; n--, j++) {
			NodeGlow* node = new_nodes_[j].parent;
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

  search_->count_propagate_node_visits_ += count;

	return count;
}



void SearchWorkerGlow::ThreadLoop(int thread_id) {

	PositionHistory history(search_->played_history_);
	std::vector<Move> movestack;

	new_nodes_ = new NewNode[new_nodes_amount_limit_];

	search_->busy_mutex_.lock();
	if (LOG_RUNNING_INFO) LOGFILE << "Working thread: " << thread_id;

	std::vector<std::mutex *> helper_thread_locks;
	std::vector<std::thread> helper_threads;
	for (int j = 0; j < std::max(N_HELPER_THREADS_PRE, N_HELPER_THREADS_POST); j++) {
		helper_thread_locks.push_back(new std::mutex());
		helper_thread_locks[j]->lock();
		std::mutex *lock = helper_thread_locks[j];
    helper_threads.emplace_back([this, j, lock]()
      {
        HelperThreadLoop(j, lock);
      }
    );
  }

	for (int n = new_nodes_amount_limit_; n > 0; n--) {
		junction_locks_.push_back(new std::mutex());
	}

//  auto board = history.Last().GetBoard();
//  if (DEBUG) LOGFILE << "Inital board:\n" << board.DebugString();

//  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

//  unsigned int lim = limits_.visits;

//  int i = 0;

  if (root_node_->GetNumEdges() == 0 && !root_node_->IsTerminal()) {  // root node not extended
    search_->ExtendNode(&history, root_node_);
    if (root_node_->IsTerminal()) {
      std::cerr << "Root " << root_node_ << " is terminal, nothing to do\n";
      abort();
    }
		//computation_ = search_->network_->NewComputation();
    computation_ = std::make_unique<CachingComputation>(std::move(search_->network_->NewComputation()),
                                                        search_->cache_);
    AddNodeToComputation(root_node_, &history);
		minibatch_shared_idx_ = 0;

    // LOGFILE << "Computing thread root ..";
    computation_->ComputeBlocking();
    // LOGFILE << " done\n";
    retrieveNNResult(root_node_, 0);
    //i++;
  }

//	auto cmp = [](PropagateQueueElement left, PropagateQueueElement right) { return left.depth < right.depth;};
//	std::priority_queue<PropagateQueueElement, std::vector<PropagateQueueElement>, decltype(cmp)> propagate_queue(cmp);


	while (search_->not_stop_searching_) {

		//computation_ = search_->network_->NewComputation();
    computation_ = std::make_unique<CachingComputation>(std::move(search_->network_->NewComputation()),
                                                        search_->cache_);

    new_nodes_amount_target_ = batch_size_;

		auto start_comp_time = std::chrono::steady_clock::now();
		auto stop_comp_time = std::chrono::steady_clock::now();

		if (OLD_PICK_N_CREATE_MODE) {
		
		helper_threads_mode_ = 1;
		//LOGFILE << "Allowing helper threads to help";
		for (int j = 0; j < N_HELPER_THREADS_PRE; j++) {
			helper_thread_locks[j]->unlock();
		}

		start_comp_time = std::chrono::steady_clock::now();
		//auto start_comp_time2 = start_comp_time;

		// LOGFILE << "Working myself.";
		pickNodesToExtend();

		// abort();		

		helper_threads_mode_ = 2;  // from now no new nodes will be added

//		if (new_nodes_.size() == 0) {  // no new nodes found, but there may exist unextended edges unavailable due to business
		if (new_nodes_size_ == 0) {  // no new nodes found, but there may exist unextended edges unavailable due to business
			for (int j = 0; j < N_HELPER_THREADS_PRE; j++) {
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

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_search_ += (stop_comp_time - start_comp_time).count();

		start_comp_time = stop_comp_time;

		buildJunctionRTree();

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_junctions_ += (stop_comp_time - start_comp_time).count();

		start_comp_time = std::chrono::steady_clock::now();

		int count = extendTree(&movestack, &history);

		if (LOG_RUNNING_INFO) LOGFILE << "main thread new nodes: " << count;

		for (int j = 0; j < N_HELPER_THREADS_PRE; j++) {
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
		//	NodeGlow* node = non_computation_new_nodes_[i].node;
		//	while (true) {
		//		node = node->GetParent();
		//		recalcPropagatedQ(node);
		//		if (node == root_node_) break;
		//	}
		//}
		//non_computation_new_nodes_.clear();

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_create_ += (stop_comp_time - start_comp_time).count();

		} else {  // new pick n create mode

			start_comp_time = std::chrono::steady_clock::now();
		picknextend(&history);
		if (new_nodes_size_ == 0) {
			if (search_->half_done_count_ == 0) {  // no other thread is waiting for nn computation and new nodes to finish so the search tree is exhausted
				search_->not_stop_searching_ = false;
				break;
			}
			search_->busy_mutex_.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			search_->busy_mutex_.lock();
			continue;
		}

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_search_ += (stop_comp_time - start_comp_time).count();

		start_comp_time = stop_comp_time;

		buildJunctionRTree();

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_junctions_ += (stop_comp_time - start_comp_time).count();

		}
		//~ if (minibatch_.size() < propagate_list_.size()) {
			//~ std::cerr << "minibatch_.size() < propagate_list_.size(): " << minibatch_.size() << " < " << propagate_list_.size() << "\n";
			//~ abort();
		//~ }

		new_nodes_list_shared_idx_ = 0;

		if (LOG_RUNNING_INFO) LOGFILE
						<< "n: " << root_node_->GetN()
//						<< ", new_nodes_ size: " << new_nodes_.size()
						<< ", new_nodes_ size: " << new_nodes_size_
            << ", new_nodes_amount_target_: " << new_nodes_amount_target_
						//<< ", minibatch_ size: " << minibatch_.size()
						<< ", junctions_ size: " << junctions_.size();
						//<< ", highest w: " << new_nodes_[new_nodes_.size() - 1].w
						//<< ", node stack size: " << nodestack_.size()
						//<< ", max_unexpanded_w: " << new_nodes_[0];

		search_->half_done_count_ += new_nodes_size_;

		//LOGFILE << "Unlock " << thread_id;
		search_->busy_mutex_.unlock();

    // std::this_thread::sleep_for(std::chrono::milliseconds(0));
		start_comp_time = std::chrono::steady_clock::now();

		if (minibatch_shared_idx_ > 0) {
			computation_->ComputeBlocking();
			search_->count_minibatch_size_ += minibatch_shared_idx_;
			minibatch_shared_idx_ = 0;
		}

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_compute_ += (stop_comp_time - start_comp_time).count();

		search_->busy_mutex_.lock();

		search_->half_done_count_ -= new_nodes_size_;

		if (LOG_RUNNING_INFO) LOGFILE << "Working thread: " << thread_id;


		//i += minibatch.size();

		start_comp_time = std::chrono::steady_clock::now();

		helper_threads_mode_ = 3;
		for (int j = 0; j < N_HELPER_THREADS_POST; j++) {
			helper_thread_locks[j]->unlock();
		}

		for (int j = 0; j < (int)new_nodes_size_; j++) {
			if (new_nodes_[j].batch_idx != -1) {
				retrieveNNResult(new_nodes_[j].new_node.get(), new_nodes_[j].batch_idx);
			}
			new_nodes_[j].parent->AddChild(std::move(new_nodes_[j].new_node));
			new_nodes_amount_retrieved_++;
		}

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_retrieve_ += (stop_comp_time - start_comp_time).count();

		helper_threads_mode_ = 4;

		start_comp_time = std::chrono::steady_clock::now();

		int pcount = propagate();

		for (int j = 0; j < N_HELPER_THREADS_POST; j++) {
			helper_thread_locks[j]->lock();
		}

		stop_comp_time = std::chrono::steady_clock::now();
		search_->duration_propagate_ += (stop_comp_time - start_comp_time).count();
		search_->count_iterations_++;

		if (LOG_RUNNING_INFO) LOGFILE << "main thread did propagates: " << pcount;

		new_nodes_list_shared_idx_ = 0;
		new_nodes_amount_retrieved_ = 0;

		search_->count_junctions_ += junctions_.size();

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
		search_->ponder_lock_.lock();
		bool ponder = search_->ponder_;
		search_->ponder_lock_.unlock();
		search_->SendUciInfo(); // Make sure uci-info is updated just before we make our move.
		if (!ponder && !search_->abort_) { // Not sure if ponder should be here
	    search_->SendMovesStats(); // Support VerboseMoveStats
			search_->reportBestMove();
		}

		int64_t elapsed_time = search_->GetTimeSinceStart();
		//LOGFILE << "Elapsed time when thread for node " << root_node_ << " which has size " << root_node_->GetN() << " nodes did " << i << " computations: " << elapsed_time << "ms";
	  if(LOG_RUNNING_INFO){
      LOGFILE << "Elapsed time for " << root_node_->GetN() << " nodes: " << elapsed_time << "ms";
      LOGFILE << "#helper threads pre: " << N_HELPER_THREADS_PRE << ", #helper threads post: " << N_HELPER_THREADS_POST;
      LOGFILE << "root Q: " << root_node_->GetQ();
      LOGFILE << "move   P                 n   norm n     Q          w";
			for (NodeGlow *i = root_node_->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
	      LOGFILE << std::fixed << std::setfill(' ') 
		      << (root_node_->GetEdges())[i->GetIndex()].move_.as_string() << " "
		      << std::setw(10) << (root_node_->GetEdges())[i->GetIndex()].GetP() << " "
		      << std::setw(10) << i->GetN() << " "
		      << std::setw(10) << (float)(i->GetN() / (float)(root_node_->GetN() - 1)) << " "
		// << std::setw(4) << (i->ComputeHeight() << " "
		      << std::setw(10) << (float)(i->GetQ()) << " "
		      << std::setw(10) << i->GetW();
	    }

      if (search_->count_iterations_ > 0) {
        int divisor = search_->count_iterations_ * 1000;
        LOGFILE << "search: " << search_->duration_search_ / divisor
                << ", junctions: " << search_->duration_junctions_ / divisor
                << ", create: " << search_->duration_create_ / divisor
                << ", compute: " << search_->duration_compute_ / divisor
                << ", retrieve: " << search_->duration_retrieve_ / divisor
                << ", propagate: " << search_->duration_propagate_ / divisor
                << ", pre: " << (search_->duration_search_ + search_->duration_junctions_ + search_->duration_create_) / divisor
                << ", post: " << (search_->duration_retrieve_ + search_->duration_propagate_) / divisor
                << ", total (exc nn comp): " << (search_->duration_search_ + search_->duration_junctions_ + search_->duration_create_ + search_->duration_retrieve_ + search_->duration_propagate_) / divisor;
      }

      if (search_->count_iterations_ > 0) {
        LOGFILE << "nodes per iteration: " << root_node_->GetN() / search_->count_iterations_
                << ", minibatch size: " << search_->count_minibatch_size_ / search_->count_iterations_
                << ", search node visits: " << search_->count_search_node_visits_ / search_->count_iterations_
                << ", propagate node visits: " << search_->count_propagate_node_visits_ / search_->count_iterations_
                << ", junctions: " << search_->count_junctions_ / search_->count_iterations_;
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


void SearchWorkerGlow::HelperThreadLoop(int helper_thread_id, std::mutex* lock) {
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
