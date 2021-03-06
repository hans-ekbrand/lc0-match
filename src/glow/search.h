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
#include "search/params.h"
#include "glow/node.h"
#include "search/search.h"
#include "neural/network.h"
#include "neural/cache.h"
#include "syzygy/syzygy.h"
#include "utils/optional.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

class SearchGlow : public SearchCommon {
public:
	SearchGlow(const NodeTreeGlow& tree, Network* network,
		 BestMoveInfo::Callback best_move_callback,
		 ThinkingInfo::Callback info_callback, const SearchLimits& limits,
		 const OptionsDict& options, NNCache* cache,
		 SyzygyTablebase* syzygy_tb,
		 bool ponder = false);

	~SearchGlow();

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
	std::vector<std::string> GetVerboseStats(NodeGlow* node, bool is_black_to_move);
	NNCacheLock GetCachedNNEval(NodeGlow* node) const;
	void reportBestMove();
	void ExtendNode(PositionHistory* history, NodeGlow* node);

	mutable std::mutex threads_list_mutex_;
	std::atomic<int> n_thread_active_{0};
	std::vector<std::thread> threads_;

	NodeGlow* root_node_;

        const int64_t initial_visits_;

	bool ponder_ = false;
	std::mutex ponder_lock_;

	std::mutex busy_mutex_;
	int iteration_count_a_ = 0;
	int iteration_count_b_ = 0;
	int half_done_count_ = 0;

	int full_tree_depth_ = 0;
	uint64_t cum_depth_ = 0;
	//std::mutex counters_lock_;

	int64_t last_uci_time_ = 0;
	std::atomic<bool> not_stop_searching_{true};
	std::atomic<bool> abort_{false};

	std::atomic<int64_t> duration_search_{0};
	std::atomic<int64_t> duration_junctions_{0};
	std::atomic<int64_t> duration_create_{0};
	std::atomic<int64_t> duration_compute_{0};
	std::atomic<int64_t> duration_retrieve_{0};
	std::atomic<int64_t> duration_propagate_{0};

  int64_t count_minibatch_size_{0};
  int64_t count_search_node_visits_{0};
  std::atomic<int64_t> count_propagate_node_visits_{0};
  int64_t count_junctions_{0};

	std::atomic<int> count_iterations_{0};

	friend class SearchWorkerGlow;
};

class SearchWorkerGlow {
public:
	SearchWorkerGlow(SearchGlow *search) :
		search_(search),
		q_concentration_(search->params_.GetCpuct()),
		p_concentration_(search->params_.GetPolicySoftmaxTemp()),
		policy_weight_exponent_(search->params_.GetCpuctFactor()),
		batch_size_(search->params_.GetMiniBatchSize()),
    new_nodes_amount_limit_(batch_size_ * 2),
		history_fill_(search->params_.GetHistoryFill()),
		played_history_length_(search_->played_history_.GetLength()),
		cache_history_length_plus_1_(search_->params_.GetCacheHistoryLength() + 1),
		root_node_(search->root_node_) { }

	void ThreadLoop(int thread_id);
	void HelperThreadLoop(int helper_thread_id, std::mutex* lock);


private:

	struct NewNode {
		NodeGlow* node;
		uint16_t junction;
		int16_t batch_idx;  // -1 if not in batch
	};

	struct Junction {
		NodeGlow *node;
		uint16_t parent;
		uint8_t children_count;
	};

	int AddNodeToComputation(NodeGlow* node, PositionHistory *history);
	void retrieveNNResult(NodeGlow* node, int batchidx);
  void recalcKnownWin(NodeGlow* node, int win_idx);
	void recalcPropagatedQ(NodeGlow* node);
	void pickNodesToExtend();
	int appendHistoryFromTo(std::vector<Move> *movestack, PositionHistory *history, NodeGlow* from, NodeGlow* to);
	float computeChildWeights(NodeGlow* node, bool evalution_weights);
	int propagate();
	int extendTree(std::vector<Move> *movestack, PositionHistory *history);
	void buildJunctionRTree();




	SearchGlow *search_;

	const float q_concentration_;
	const float p_concentration_;
	const float policy_weight_exponent_; // weight of policy relative to weight of q: pow(n, pwe)/n where n is the number of subnodes of the current node.	
	const int batch_size_;
  const int new_nodes_amount_limit_;
	const FillEmptyHistory history_fill_;
	const int played_history_length_;
  const int cache_history_length_plus_1_;

	NodeGlow* root_node_;


	//std::unique_ptr<NetworkComputation> computation_;
	std::unique_ptr<CachingComputation> computation_;
	std::mutex computation_lock_;  // SearchWorker instance not needed, move to Search?

	//std::vector<NewNode> new_nodes_;
	NewNode *new_nodes_;
	std::atomic<int> new_nodes_size_{0};
  std::atomic<int> new_nodes_amount_target_{0};
	int new_nodes_list_shared_idx_ = 0;  // SearchWorker instance not needed, move to Search?
	std::mutex new_nodes_list_lock_;  // SearchWorker instance not needed, move to Search?

	std::vector<Junction> junctions_;
	std::vector<std::mutex *> junction_locks_;  // SearchWorker instance not needed, move to Search?

	int minibatch_shared_idx_ = 0;
	std::atomic<int> new_nodes_amount_retrieved_{0};  // SearchWorker instance not needed, move to Search?

	std::vector<float> pvals_;  // SearchWorker instance not needed, move to Search?

	int helper_threads_mode_ = 0;  // SearchWorker instance not needed, move to Search?

};


}  // namespace lczero
