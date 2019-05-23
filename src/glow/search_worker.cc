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

//#include <signal.h>

#include "glow/search.h"
#include "glow/search_worker.h"
#include "glow/strategy.h"

#include <iostream>
#include <fstream>
#include <random> // normal distribution
#include <math.h>
#include <cassert>  // assert() used for debugging during development
#include <iomanip>

#include "neural/encoder.h"


namespace lczero {

namespace {

// Alternatives:

int const MAX_NEW_SIBLINGS = 10000;
  // The maximum number of new siblings. If 1, then it's like old MULTIPLE_NEW_SIBLINGS = false, if >= maximum_number_of_legal_moves it's like MULTIPLE_NEW_SIBLINGS = true
const int kUciInfoMinimumFrequencyMs = 5000;

int const N_HELPER_THREADS_PRE = 3;
int const N_HELPER_THREADS_POST = 3;

bool const DEBUG_MODE = false;

bool const OLD_PICK_N_CREATE_MODE = true;

// int debug_state[4];
// 
// void segfault_sigaction(int signal, siginfo_t *si, void *arg)
// {
//     printf("Caught segfault at address %p\n", si->si_addr);
// 		for (int i = 0; i < 4; i++) {
// 			std::cerr << debug_state[i] << "\n";
// 		}
//     exit(0);
// }


}  // namespace




void SearchWorkerGlow::pickNodesToExtend() {
	NodeGlow* node;
	NodeGlow *best_child;

	int nodes_visited = 0;
	std::random_device rd{};
	std::mt19937 gen{rd()};

	for (int n = 0; n < new_nodes_amount_target_; n++) {
//std::cout << root_node_->GetMaxW();

		node = root_node_;

		int depth = 0;

		bool set_bestchild = true;

		while (true) {
			nodes_visited++;
			if(set_bestchild){
			  best_child = node->GetBestChild();
			} else {
			  set_bestchild = true;
			}

			if(best_child != nullptr){
			  // If best_child is not the child with highest Q, then gamble against the child with highest Q. If you loose, choose child with highest q.
			  float maxq = -2.0;
			  NodeGlow* child_with_highest_q;
			  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
			    float q = i->GetQ();
			    if (q > maxq){
			      maxq = q;
			      child_with_highest_q = i;
			    }
			  }
			  if(best_child->GetQ() != maxq && ! child_with_highest_q->IsTerminal()){ // added test for IsTerminal to get rid of segfaults at gpu-master.
			    // Sample a value from a normal distribution with maxq as mean. If our q is better than that, then we can continue.
			    std::normal_distribution<> d{maxq, 0.044}; // 0.021 = sqrt(0.044)
			    if(d(gen) < best_child->GetQ()){
			      // Let's encourage exploration by giving a random child of root a (low) probability of being entered at this point. Note that starting at certain child does not ensure that it will be entered, it still has
			      // to pass the pruning test, or the highest q of root will be entered.
			      // Actually we can use a constant here because with increasing depth, each additional node that is to be extended will have to pass more and more tests which will increase the probability of ending up here.
			      float p_for_starting_at_random_root_child = 0.03;
			      std::uniform_real_distribution<> dist_uni_real(0, 1);
			      float my_sample = dist_uni_real(gen);
			      if(my_sample < p_for_starting_at_random_root_child){
				// make some random child of root best_child
				node = root_node_;
				std::uniform_int_distribution<int> dist_uni_int(1, node->GetNumEdges());
				int this_one = dist_uni_int(gen);
				for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
				  if(i->GetIndex() == this_one){
				    best_child = i;
				    set_bestchild = false;
				    continue;
				  }
				}
			      } else {
				best_child = child_with_highest_q;
			      }
			    }
			  }
			} else {
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

			update_junctions(node, junction_mode, ccidx);

			if (node == root_node_) break;
			node = node->GetParent();
		}
	}

  search_->count_search_node_visits_ += nodes_visited;
}

inline void SearchWorkerGlow::update_junctions(NodeGlow *node, int &junction_mode, uint16_t &ccidx) {
	if (junction_mode == 0) {
		uint16_t n = node->GetBranchingInFlight();
		if (n == 0) {  // an unvisited node
			node->SetBranchingInFlight(ccidx);
		} else if (n & 0xC000) {  // part of a path between junctions
			uint16_t new_junc_idx = junctions_size_++;
			if (new_junc_idx >= new_nodes_amount_limit_) {
				std::cerr << "Too many junctions\n";
				abort();
			}
			node->SetBranchingInFlight(new_junc_idx + 1);
			uint16_t parent;
			if (n & 0x8000) {
				parent = new_nodes_[n & 0x3FFF].junction;
				new_nodes_[n & 0x3FFF].junction = new_junc_idx;
			} else {
				parent = junctions_[n & 0x3FFF].parent;
				junctions_[n & 0x3FFF].parent = new_junc_idx;
			}
			junctions_[new_junc_idx] = {node, parent, 0};

			if (ccidx & 0x8000) {
				new_nodes_[ccidx & 0x3FFF].junction = new_junc_idx;
			} else {  // never the case?
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
}

  // makes a junction at a node which is on a path but not previously on a junction
void SearchWorkerGlow::insert_junction(NodeGlow *node) {
	uint16_t n = node->GetBranchingInFlight();
	uint16_t new_junc_idx = junctions_size_++;
	if (new_junc_idx >= new_nodes_amount_limit_) {
		std::cerr << "Too many junctions\n";
		abort();
	}
	node->SetBranchingInFlight(new_junc_idx + 1);
	uint16_t parent;
	if (n & 0x8000) {
		parent = new_nodes_[n & 0x3FFF].junction;
		new_nodes_[n & 0x3FFF].junction = new_junc_idx;
	} else {
		parent = junctions_[n & 0x3FFF].parent;
		junctions_[n & 0x3FFF].parent = new_junc_idx;
	}
	junctions_[new_junc_idx] = {node, parent, 0};

	uint16_t ccidx = new_junc_idx | 0x4000;
	for (;;) {
		if (node == root_node_) break;
		node = node->GetParent();
		if ((node->GetBranchingInFlight() & 0xC000) == 0) break;
		node->SetBranchingInFlight(ccidx);
	}
	
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
    if (new_nodes_amount_target_ < new_nodes_amount_limit_) new_nodes_amount_target_++;  // it's cached so it shouldn't be counted towards the minibatch size
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
	computation_lock_.lock();
	{
		NNCacheLock nneval(search_->cache_, hash);
		if (nneval) {  // it's cached
			
			//float q = -computation_->GetQVal(batchidx);
			float q = -nneval->q;
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
				if (nneval->p[k].first != node->GetEdges()[k].move_.as_nn_index()) {std::cerr << "as_nn_index mismatch\n"; abort();};
				float p = nneval->p[k].second;
				if (p < 0.0) {
					std::cerr << "p value < 0\n";
					abort();
					//p = 0.0;
				}
				//if (p > 1.0) {
				//  std::cerr << "p value > 1\n";
				//  abort();
				//}
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

			computation_lock_.unlock();
			return -1;
		}
	}
	computation_lock_.unlock();

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
					if (new_nodes_amount_target_ < new_nodes_amount_limit_) new_nodes_amount_target_++;  // it's cached so it shouldn't be counted towards the minibatch size
				} else {
					new_nodes_[i].batch_idx = idx;
				}
			} else {  // is terminal
				if (new_nodes_amount_target_ < new_nodes_amount_limit_) new_nodes_amount_target_++;  // it's terminal so it shouldn't be counted towards the minibatch size
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


int const MAX_SUBTREE_SIZE = 1000;  // 50;  // 100000000;  // 100;
int const LOCAL_NODES_AMOUNT = 1;  // 20;  // 1;  // 10;
//float const OVERLAP_FACTOR = 10.0;

void SearchWorkerGlow::picknextend_reference(std::vector<Move> *movestack, PositionHistory *history) {

	int nodes_visited = 0;
	int full_tree_depth = search_->full_tree_depth_;
	int cum_depth = 0;

	while (new_nodes_size_ < new_nodes_amount_target_) {  // repeat this until minibatch_size amount of non terminal, non cache hit nodes have been found (or reached a predefined limit larger than minibatch size)
		NodeGlow *node = root_node_;
		if (node->GetMaxW() == 0.0) break;  // no more expandable node
		NodeGlow *best_child = node->GetBestChild();
		
		while (true) {
			if (best_child == nullptr || node->GetN() <= MAX_SUBTREE_SIZE) break;
			node = best_child;
			best_child = node->GetBestChild();
		}
		
		NodeGlow *local_root = node;
		
		int n_local_nodes = std::min(LOCAL_NODES_AMOUNT, new_nodes_amount_target_ - new_nodes_size_);

		
		for (; n_local_nodes > 0; n_local_nodes--) {  // repeat until localbatch_size amount of nodes have been found
			// go down to max unexpanded node
			while (true) {
				if (best_child == nullptr) break;  // best unexpanded node is child of this node
				node = best_child;
				best_child = node->GetBestChild();
			};

			int nidx = node->GetNextUnexpandedEdge();
			node->SetNextUnexpandedEdge(nidx + 1);

			std::unique_ptr<NodeGlow>newnode = std::make_unique<NodeGlow>(node, nidx);

			bool out_of_order = false;
			int nnidx = -1;

			appendHistoryFromTo(movestack, history, root_node_, newnode.get());

			search_->ExtendNode(history, newnode.get());
			if (newnode.get()->IsTerminal()) {
				out_of_order = true;
				node->AddChild(std::move(newnode));
			} else {
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
					nodes_visited++;
					if (node == root_node_) break;
					node = node->GetParent();
				}
			} else {  // not out of order
				int junction_mode = 0;
				uint16_t ccidx = nnidx | 0x8000;

				while (true) {
					recalcMaxW(node);
					nodes_visited++;

					update_junctions(node, junction_mode, ccidx);

					if (node == root_node_) break;
					node = node->GetParent();
				}
			}

			node = local_root;
			if (node->GetMaxW() == 0.0) break;  // no more expandable node in sub tree
			best_child = node->GetBestChild();
		}

	}

// 	history->Trim(played_history_length_);

	if (full_tree_depth > search_->full_tree_depth_) search_->full_tree_depth_ = full_tree_depth;
	search_->cum_depth_ += cum_depth;

	search_->count_search_node_visits_ += nodes_visited;

}


void SearchWorkerGlow::picknextend(PositionHistory *history) {

	n_searchers_active_++;
	//int debug_state_idx = n_searchers_active_++;

	tree_lock_.lock();

//	std::vector<int> path;
//	unsigned int last_depth = 0;
	int nodes_visited = 0;
//	bool same_path = false;
	int full_tree_depth = search_->full_tree_depth_;
	int cum_depth = 0;

	
	// turn on global tree lock
	while (new_nodes_size_ <= new_nodes_amount_target_ - n_searchers_active_) {  // repeat this until minibatch_size amount of non terminal, non cache hit nodes have been found (or reached a predefined limit larger than minibatch size)

		NodeGlow *node = root_node_;
		if (node->GetMaxW() == 0.0) break;  // no more expandable node
		NodeGlow *best_child = node->GetBestChild();
//		if (best_child == nullptr && (node->GetNextUnexpandedEdge() == node->GetNumEdges() || node->GetNextUnexpandedEdge() - node->GetNumChildren() == MAX_NEW_SIBLINGS)) break;  // no more expandable node
		
		while (true) {
			if (best_child == nullptr || node->GetN() <= MAX_SUBTREE_SIZE) break;
			history->Append(node->GetEdges()[best_child->GetIndex()].move_);
			node = best_child;
			best_child = node->GetBestChild();
		}

		if (node->GetN() > MAX_SUBTREE_SIZE) {

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
					nodes_visited++;
					if (node == root_node_) break;
					node = node->GetParent();
				}
			} else {  // not out of order
				int junction_mode = 0;
				uint16_t ccidx = nnidx | 0x8000;

				while (true) {
					recalcMaxW(node);
					nodes_visited++;

					update_junctions(node, junction_mode, ccidx);

					if (node == root_node_) break;
					node = node->GetParent();
				}
			}
			
			continue;
		}

		NodeGlow *local_root = node;

		if (node->GetBranchingInFlight() & 0xC000) {
			insert_junction(node);
		}
		
//		float old_global_maxw = root_node_->GetMaxW();
//		float old_local_maxw = local_root->GetMaxW();
		node->SetMaxW(0.0);
		while (node != root_node_) {
			node = node->GetParent();

			recalcMaxW(node);
		}
//		float new_global_maxw_local_equiv = (old_global_maxw - OVERLAP_FACTOR * (old_global_maxw - root_node_->GetMaxW())) * old_local_maxw / old_global_maxw;
//std::cerr << old_local_maxw << " " << old_global_maxw << " " << root_node_->GetMaxW() << " " << new_global_maxw_local_equiv << " ";

		tree_lock_.unlock();

		node = local_root;

		int local_root_depth = history->GetLength();

//std::cerr << "d: " << local_root_depth - played_history_length_ << "\n";
		
		int global_junction_mode = 2;
		uint16_t global_ccidx = 0;  // value unimportant
		
		bool global_out_of_order = false;
		
		// starting from root node follow maxidx until next move would make the sub tree to small
		// propagate no availability upwards to root
		// turn off global tree lock

//int did = 0;

		NQMaxw local_prop({0, 0.0, 0.0});  // don't care

		for (int n = LOCAL_NODES_AMOUNT; n > 0; n--) {

//did++;
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
// 				}
				node = best_child;
				best_child = node->GetBestChild();
//				depth++;
			};

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
			history->Trim(local_root_depth);

			if (depth > full_tree_depth) full_tree_depth = depth;
			cum_depth += depth;

			if (out_of_order) {
				global_out_of_order = true;
				while (true) {
					local_prop = recalcPropagatedQ_local(node);
					nodes_visited++;
					if (node == local_root) break;
					node->SetN(local_prop.n);
					node->SetQ(local_prop.q);
					node->SetMaxW(local_prop.maxw);
					node = node->GetParent();
				}
			} else {  // not out of order
				int junction_mode = 0;
				uint16_t ccidx = nnidx | 0x8000;

				while (true) {
					local_prop.maxw = recalcMaxW_local(node);
					nodes_visited++;

					update_junctions(node, junction_mode, ccidx);

					if (node == local_root) break;
					node->SetMaxW(local_prop.maxw);
					node = node->GetParent();
				}
				
				global_junction_mode = std::min(global_junction_mode, junction_mode);
				if (junction_mode < 2) global_ccidx = ccidx;
			}

			if (local_prop.maxw == 0.0) break;  // no more expandable node in sub tree
//			if (local_prop.maxw < new_global_maxw_local_equiv) break;

			if (new_nodes_size_ > new_nodes_amount_target_ - n_searchers_active_) break;

			// node = local_root
			best_child = node->GetBestChild();
			
			// when deviating from previous path, trim history according to this and start pushing each new move
			// create node, increment inner loop node count
			// compute legal moves
			// if node is terminal or cache hit, set its q (and p:s) and turn on full propagation mode
			// otherwise, add node to computation list and turn on limited max only propagation and forking tree mode. Increment computation node count.
			// propagate to local tree root according to mode
		}
//std::cerr << "did " << did << "\n";

		tree_lock_.lock();

		if (global_out_of_order) {
			node->SetN(local_prop.n);
			node->SetQ(local_prop.q);
		}
		node->SetMaxW(local_prop.maxw);

		history->Trim(played_history_length_);

		while (node != root_node_) {
			node = node->GetParent();

			if (global_out_of_order) {
				recalcPropagatedQ(node);
			} else {
				recalcMaxW(node);
			}
			nodes_visited++;
			
			update_junctions(node, global_junction_mode, global_ccidx);
		}

		// turn on global tree lock
		// propagate to root
	}

// 	history->Trim(played_history_length_);

	if (full_tree_depth > search_->full_tree_depth_) search_->full_tree_depth_ = full_tree_depth;
	search_->cum_depth_ += cum_depth;

	search_->count_search_node_visits_ += nodes_visited;

	tree_lock_.unlock();

	n_searchers_active_--;

	
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
    if (p < 0.0) {
      std::cerr << "p value < 0\n";
      abort();
      //p = 0.0;
    }
    //if (p > 1.0) {
    //  std::cerr << "p value > 1\n";
    //  abort();
    //}
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


inline float SearchWorkerGlow::recalcMaxW_local(NodeGlow *node) {
	NodeGlow *max_child = nullptr;
	float max_w = 0.0;
	int nidx = node->GetNextUnexpandedEdge();
	if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
	  if (node->GetN() > search_->params_.GetMaxCollisionVisitsId()){
	    int depth = 0;
	    NodeGlow *node_tmp = node;
	    while (node_tmp != nullptr) {
	      node_tmp = node_tmp->GetParent();
	      depth++;
	    }
	    float coeff = 1.0;
	    coeff = coeff * pow(search_->params_.GetCpuct(), log2(depth));
	    max_w = node->GetEdges()[nidx].GetP() * coeff;
	  } else {
	    max_w = node->GetEdges()[nidx].GetP();
	  }
	}
	for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
		float br_max_w = i->GetW() * i->GetMaxW();
		if (br_max_w > max_w) {
			max_w = br_max_w;
			max_child = i;
		}
	}
	node->SetBestChild(max_child);
	return max_w;
}

inline void SearchWorkerGlow::recalcMaxW(NodeGlow *node) {
	node->SetMaxW(recalcMaxW_local(node));
}



inline int recalcN(NodeGlow *node) {
  int n = 1;
	for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
		n += i->GetN();
	}
	return n;
}

// returns n or -n if node has turned terminal
inline float checkTerminal(NodeGlow* node) {
	if (node->GetNumChildren() < node->GetNumEdges()) {
		for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
			if (i->IsTerminal()) {
				if (i->GetQ() == 1.0) {
					return -1.0;
				}
			}
		}
		return 2.0;
	} else {
		float max_q_terminal = -1.0;
		for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
			if (i->IsTerminal()) {
				float const q = i->GetQ();
				if (q == 1.0) {
					return -1.0;
				} else {
					if (q > max_q_terminal) max_q_terminal = q;
				}
			} else {
				for (i = i->GetNextSibling(); i != nullptr; i = i->GetNextSibling()) {
					if (i->IsTerminal()) {
						if (i->GetQ() == 1.0) {
							return -1.0;
						}
					}
				}
				return 2.0;
			}
		}
		// all children present, all terminal, non with q = 1
		
		return -max_q_terminal;
	}
}

void SearchWorkerGlow::recalcPropagatedQ(NodeGlow* node) {
	int n = recalcN(node);
	float termq = checkTerminal(node);
	if (termq < 2.0) {  // terminal
		node->SetN(n);
		node->SetQ(termq);
		node->SetBestChild(nullptr);
		node->SetMaxW(0.0);
//		node->is_terminal_ = true;
		return;
	}
  node->SetN(n);

	float q = compute_q_and_weights(node, n);
// 	if (isnan(q)) {
// 		std::cerr << "q is nan\n";
// 		abort();
// 	}
	node->SetQ(q);

	recalcMaxW(node);
}

SearchWorkerGlow::NQMaxw SearchWorkerGlow::recalcPropagatedQ_local(NodeGlow* node) {
	int n = recalcN(node);
	float termq = checkTerminal(node);
	if (termq < 2.0) {  // terminal
		node->SetBestChild(nullptr);
//		node->is_terminal_ = true;
		return {n, termq, 0.0};
	}

	float q = compute_q_and_weights(node, n);
// 	if (isnan(q)) {
// 		std::cerr << "q is nan\n";
// 		abort();
// 	}

	float maxw = recalcMaxW_local(node);
	return {n, q, maxw};
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

// 	for (int i = 0; i < 4; i++) {
// 		debug_state[i] = -1;
// 	}
// 	struct sigaction sa;
// 	memset(&sa, 0, sizeof(struct sigaction));
// 	sigemptyset(&sa.sa_mask);
// 	sa.sa_sigaction = segfault_sigaction;
// 	sa.sa_flags   = SA_SIGINFO;
// 	sigaction(SIGSEGV, &sa, NULL);
	
	
	PositionHistory history(search_->played_history_);
	std::vector<Move> movestack;

	new_nodes_ = new NewNode[new_nodes_amount_limit_];

	search_->busy_mutex_.lock();
	//if (DEBUG_MODE) LOGFILE << "Working thread: " << thread_id;

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

  junctions_ = new Junction[new_nodes_amount_limit_];
	junction_locks_ = new std::mutex *[new_nodes_amount_limit_];
	for (int n = new_nodes_amount_limit_ - 1; n >= 0; n--) {
		junction_locks_[n] = new std::mutex();
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

		//LOGFILE << "Working myself.";
    pickNodesToExtend();

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

		//if (DEBUG_MODE) LOGFILE << "main thread new nodes: " << count;

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

		helper_threads_mode_ = 1;
		//LOGFILE << "Allowing helper threads to help";
		for (int j = 0; j < N_HELPER_THREADS_PRE; j++) {
			helper_thread_locks[j]->unlock();
		}

		start_comp_time = std::chrono::steady_clock::now();

		picknextend(&history);
//		picknextend_reference(&movestack, &history);

		for (int j = 0; j < N_HELPER_THREADS_PRE; j++) {
			helper_thread_locks[j]->lock();
		}

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

		}  // end new pick n create mode

		//~ if (minibatch_.size() < propagate_list_.size()) {
			//~ std::cerr << "minibatch_.size() < propagate_list_.size(): " << minibatch_.size() << " < " << propagate_list_.size() << "\n";
			//~ abort();
		//~ }

		new_nodes_list_shared_idx_ = 0;

// 		if (DEBUG_MODE) LOGFILE
// 						<< "n: " << root_node_->GetN()
// //						<< ", new_nodes_ size: " << new_nodes_.size()
// 						<< ", new_nodes_ size: " << new_nodes_size_
//             << ", new_nodes_amount_target_: " << new_nodes_amount_target_
// 						//<< ", minibatch_ size: " << minibatch_.size()
// 						<< ", junctions_ size: " << junctions_size_;
// 						//<< ", highest w: " << new_nodes_[new_nodes_.size() - 1].w
// 						//<< ", node stack size: " << nodestack_.size()
// 						//<< ", max_unexpanded_w: " << new_nodes_[0];

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

		//if (DEBUG_MODE) LOGFILE << "Working thread: " << thread_id;


		//i += minibatch.size();

// float old_root_q = root_node_->GetQ();

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

// std::cout << " " << abs(old_root_q - root_node_->GetQ()) << "\n";

		//if (DEBUG_MODE) LOGFILE << "main thread did propagates: " << pcount;

		new_nodes_list_shared_idx_ = 0;
		new_nodes_amount_retrieved_ = 0;

    search_->count_junctions_ += junctions_size_;

		//junctions_.clear();
		junctions_size_ = 0;

		//new_nodes_.clear();
		new_nodes_size_ = 0;
		
//		root_node_->checkTree(&history);
//		std::cerr << "checkTree ok\n";

		int64_t time = search_->GetTimeSinceStart();
		if (time - search_->last_uci_time_ > kUciInfoMinimumFrequencyMs) {
			search_->last_uci_time_ = time;
			if (!DEBUG_MODE) search_->SendUciInfo();
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
	  if (DEBUG_MODE){
      LOGFILE << "Elapsed time for " << root_node_->GetN() << " nodes: " << elapsed_time << "ms";
      LOGFILE << "#helper threads pre: " << N_HELPER_THREADS_PRE << ", #helper threads post: " << N_HELPER_THREADS_POST;

			const bool is_black_to_move = search_->played_history_.IsBlackToMove();
			root_node_->show(is_black_to_move);
			NodeGlow *bestmovenode = search_->indexOfHighestQEdge(root_node_, is_black_to_move, true);
			bestmovenode->show(!is_black_to_move);

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

      if(DEBUG_MODE) root_node_->depthProfile();
      if(DEBUG_MODE) root_node_->princVarProfile();
    }
	}

	for (int i = 0; i < new_nodes_amount_limit_; i++) {
		delete junction_locks_[i];
	}
// 	while (!junction_locks_.empty()) {
// 		delete junction_locks_.back();
// 		junction_locks_.pop_back();
// 	}
	delete[] junction_locks_;
	delete[] junctions_;

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
			if (OLD_PICK_N_CREATE_MODE) {
				int count = extendTree(&movestack, &history);
				//if (DEBUG_MODE) if (count > 0) LOGFILE << "helper thread " << helper_thread_id << " did new nodes: " << count;
			} else {
				picknextend(&history);
			}
		} else {
			if (helper_threads_mode_ == 3 || helper_threads_mode_ == 4) {
				int count = propagate();
				//if (DEBUG_MODE) LOGFILE << "helper thread " << helper_thread_id << " did propagates: " << count;
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
