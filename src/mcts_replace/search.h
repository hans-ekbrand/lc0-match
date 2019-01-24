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

#pragma once

#include <functional>
#include <thread>
#include <mutex>
#include <queue>
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
	std::int64_t visits = 4000000000;
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
		 SyzygyTablebase* syzygy_tb);

	~Search_revamp();

	/* // Populates UciOptions with search parameters. */
	/* static void PopulateUciParams(OptionsParser* options); */

	// Starts worker threads and returns immediately.
	void StartThreads(size_t how_many);

	// Starts search with k threads and wait until it finishes.
	//  void RunBlocking(size_t threads);

	// Stops search. At the end bestmove will be returned. The function is not
	// blocking, so it returns before search is actually done.
	void Stop();
	// Stops search, but does not return bestmove. The function is not blocking.
	//  void Abort();
	// Blocks until all worker thread finish.
	void Wait();
	// Returns whether search is active. Workers check that to see whether another
	// search iteration is needed.
	//  bool IsSearchActive() const;

	// Returns best move, from the point of view of white player. And also ponder.
	// May or may not use temperature, according to the settings.
	//  std::pair<Move, Move> GetBestMove() const;
	// Returns the evaluation of the best move, WITHOUT temperature. This differs
	// from the above function; with temperature enabled, these two functions may
	// return results from different possible moves.
	//  float GetBestEval() const;

	// Strings for UCI params. So that others can override defaults.
	// TODO(mooskagh) There are too many options for now. Factor out that into a
	// separate class.

private:
//	void WatchdogThread();

	int64_t GetTimeSinceStart() const;

	std::mutex threads_list_mutex_;
	int n_thread_active_ = 0;
	std::vector<std::thread> threads_;

	int helper_threads_mode_ = 0;


	Node_revamp* root_node_;

	// Fixed positions which happened before the search.
	const PositionHistory& played_history_;

	Network* const network_;
	const SearchLimits_revamp limits_;

	const SearchParams params_;

	const std::chrono::steady_clock::time_point start_time_;
	const int64_t initial_visits_;

	BestMoveInfo::Callback best_move_callback_;
	ThinkingInfo::Callback info_callback_;
	// External parameters
	//  const int kMiniBatchSize;
	//const int kCacheHistoryLength;



	void AddNodeToComputation(PositionHistory *history);
	void retrieveNNResult(Node_revamp* node, int batchidx);
	void recalcPropagatedQ(Node_revamp* node);
	void pickNodesToExtend(Node_revamp* current_node, float global_weight);
	void pushNewNodeCandidate(float w, Node_revamp* node, int idx);
	int appendHistoryFromTo(std::vector<Move> *movestack, PositionHistory *history, Node_revamp* from, Node_revamp* to);
	float computeChildWeights(Node_revamp* node);
	std::vector<float> q_to_prob(std::vector<float> Q, int depth, float multiplier, float max_focus);
	void SendUciInfo();
	void ThreadLoop(int thread_id);
	void HelperThreadLoop(int helper_thread_id, std::mutex* lock);

	std::mutex busy_mutex_;

	std::vector<float> pvals_;
	//std::vector<Node_revamp *> nodestack_;
	std::vector<float> global_weight_stack_;

	struct NewNodeCandidate {
		float w;
		Node_revamp* node;
		int idx;
	};

	std::vector<struct NewNodeCandidate> node_prio_queue_;
  int node_prio_queue_nextelt_ = 0;
  std::mutex node_prio_queue_lock_;

	std::unique_ptr<NetworkComputation> computation_;
	std::mutex computation_lock_;

	std::vector<Node_revamp *> minibatch_;
	std::mutex minibatch_lock_;

	struct PropagateQueueElement {
		int depth;
		Node_revamp* node;
	};

	std::vector<PropagateQueueElement> propagate_list_;
	std::mutex propagate_list_lock_;

	int full_tree_depth_ = 0;
	uint64_t cum_depth_ = 0;
	std::mutex counters_lock_;

	//std::mutex qq_lock_;

	float q_concentration_;
	float p_concentration_;


	int64_t last_uci_time_ = 0;

	int64_t duration_search_ = 0;
	int64_t duration_create_ = 0;
	int64_t duration_compute_ = 0;
	int64_t duration_retrieve_ = 0;
	int64_t duration_propagate_ = 0;

	int64_t duration_node_prio_queue_lock_ = 0;

	int count_iterations_ = 0;

};



}  // namespace lczero
