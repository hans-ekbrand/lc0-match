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

#pragma once

#include <functional>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/params.h"
#include "mcts_replace/node.h"
#include "neural/network.h"
#include "neural/cache.h"
#include "syzygy/syzygy.h"
#include "utils/optional.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

struct SearchLimits_revamp {
	// Type for N in nodes is currently uint32_t, so set limit in order not to
	// overflow it.
	std::int64_t visits = -1;  // Why is this 4000000000 in mcts code?
	std::int64_t playouts = -1;
	int depth = -1;
	optional<std::chrono::steady_clock::time_point> search_deadline;
	bool infinite = false;
	MoveList searchmoves;

	std::string DebugString() const;
};

class Search_revamp {
public:
	Search_revamp(const NodeTree_revamp& tree, Network* network,
		 BestMoveInfo::Callback best_move_callback,
		 ThinkingInfo::Callback info_callback, const SearchLimits_revamp& limits,
		 const OptionsDict& options, NNCache* cache,
		 SyzygyTablebase* syzygy_tb,
		 bool ponder = false);

	~Search_revamp();

	// Starts worker threads and returns immediately.
	void StartThreads(size_t how_many);

	// Starts search with k threads and wait until it finishes.
	void RunBlocking(size_t threads);

	// Stops search. At the end bestmove will be returned. The function is not
	// blocking, so it returns before search is actually done.
	void Stop();
	// Stops search, but does not return bestmove. The function is not blocking.
	void Abort();
	// Blocks until all worker thread finish.
	void Wait();
	// Returns whether search is active. Workers check that to see whether another
	// search iteration is needed.
	bool IsSearchActive() const;

private:

	int64_t GetTimeSinceStart() const;
  int64_t GetTimeToDeadline() const;
	void SendUciInfo();
	void checkLimitsAndMaybeTriggerStop();
	void SendMovesStats();
	std::vector<std::string> GetVerboseStats(Node_revamp* node, bool is_black_to_move);
	NNCacheLock GetCachedNNEval(Node_revamp* node) const;
	void reportBestMove();
	void ExtendNode(PositionHistory* history, Node_revamp* node);

	mutable std::mutex threads_list_mutex_;
	std::atomic<int> n_thread_active_{0};
	std::vector<std::thread> threads_;

	Node_revamp* root_node_;
	NNCache* cache_;
	SyzygyTablebase* syzygy_tb_;
	// Fixed positions which happened before the search.
	const PositionHistory& played_history_;

	Network* const network_;
	const SearchLimits_revamp limits_;

	const SearchParams params_;

	bool ponder_ = false;
	std::mutex ponder_lock_;

	const std::chrono::steady_clock::time_point start_time_;
	const int64_t initial_visits_;

	BestMoveInfo::Callback best_move_callback_;
	ThinkingInfo::Callback info_callback_;

	std::mutex busy_mutex_;
	int iteration_count_a_ = 0;
	int iteration_count_b_ = 0;
	int half_done_count_ = 0;

	int full_tree_depth_ = 0;
	uint64_t cum_depth_ = 0;
	//std::mutex counters_lock_;
  std::atomic<int> tb_hits_{0};


	int64_t last_uci_time_ = 0;
	std::atomic<bool> not_stop_searching_{true};
	std::atomic<bool> abort_{false};

	std::atomic<int64_t> duration_search_{0};
	std::atomic<int64_t> duration_create_{0};
	std::atomic<int64_t> duration_compute_{0};
	std::atomic<int64_t> duration_retrieve_{0};
	std::atomic<int64_t> duration_propagate_{0};
	std::atomic<int> count_iterations_{0};

	friend class SearchWorker_revamp;
};

class SearchWorker_revamp {
public:
	SearchWorker_revamp(Search_revamp *search) :
		search_(search),
		q_concentration_(search->params_.GetCpuct()),
		p_concentration_(search->params_.GetPolicySoftmaxTemp()),
		policy_weight_exponent_(search->params_.GetFpuValue()),
		batch_size_(search->params_.GetMiniBatchSize()),
		history_fill_(search->params_.GetHistoryFill()),
		played_history_length_(search_->played_history_.GetLength()),
		cache_history_length_plus_1_(search_->params_.GetCacheHistoryLength() + 1),
		root_node_(search->root_node_) {}

	void ThreadLoop(int thread_id);
	void HelperThreadLoop(int helper_thread_id, std::mutex* lock);


private:

	struct NewNode {
		Node_revamp* node;
		uint16_t junction;
		int16_t batch_idx;  // -1 if not in batch
	};

	struct Junction {
		Node_revamp *node;
		uint16_t parent;
		uint8_t children_count;
	};

	int AddNodeToComputation(Node_revamp* node, PositionHistory *history);
	void retrieveNNResult(Node_revamp* node, int batchidx);
	void recalcPropagatedQ(Node_revamp* node);
	void pickNodesToExtend();
	int appendHistoryFromTo(std::vector<Move> *movestack, PositionHistory *history, Node_revamp* from, Node_revamp* to);
	float computeChildWeights(Node_revamp* node);
	int propagate();
	int extendTree(std::vector<Move> *movestack, PositionHistory *history);
	void buildJunctionRTree();




	Search_revamp *search_;

	const float q_concentration_;
	const float p_concentration_;
	const float policy_weight_exponent_; // weight of policy relative to weight of q: pow(n, pwe)/n where n is the number of subnodes of the current node.	
	const int batch_size_;
	const FillEmptyHistory history_fill_;
	const int played_history_length_;
  const int cache_history_length_plus_1_;

	Node_revamp* root_node_;


	//std::unique_ptr<NetworkComputation> computation_;
	std::unique_ptr<CachingComputation> computation_;
	std::mutex computation_lock_;  // SearchWorker instance not needed, move to Search?

	//std::vector<NewNode> new_nodes_;
	NewNode *new_nodes_;
	std::atomic<int> new_nodes_size_{0};
	int new_nodes_list_shared_idx_ = 0;  // SearchWorker instance not needed, move to Search?
	std::mutex new_nodes_list_lock_;  // SearchWorker instance not needed, move to Search?

	std::unordered_map<Node_revamp*, uint16_t> junction_of_node_;  // SearchWorker instance not needed, move to Search?

	std::vector<Junction> junctions_;
	std::vector<std::mutex *> junction_locks_;  // SearchWorker instance not needed, move to Search?

	int minibatch_shared_idx_ = 0;
	std::atomic<int> new_nodes_amount_retrieved_{0};  // SearchWorker instance not needed, move to Search?

	std::vector<float> pvals_;  // SearchWorker instance not needed, move to Search?

	int helper_threads_mode_ = 0;  // SearchWorker instance not needed, move to Search?

};


}  // namespace lczero
