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
#include <shared_mutex>
#include <thread>
#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/params.h"
#include "mcts_replace/node.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "syzygy/syzygy.h"
#include "utils/mutex.h"
#include "utils/optional.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

extern std::atomic<int> full_tree_depth;

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
  void WatchdogThread();

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  Node_revamp* root_node_;

  NNCache* cache_;

  // Fixed positions which happened before the search.
  /*const*/ PositionHistory/*&*/ played_history_;  // not const ref so that startthreads can create top of tree so that there is a leaf for each thread

  Network* const network_;
  const SearchLimits_revamp limits_;

  const SearchParams params_;

  BestMoveInfo::Callback best_move_callback_;
  // External parameters
  //  const int kMiniBatchSize;
  //const int kCacheHistoryLength;

  friend class SearchWorker_revamp;

  void AddNodeToComputation(Node_revamp* node);
};

// Single thread worker of the search engine.
// That used to be just a function Search::Worker(), but to parallelize it
// within one thread, have to split into stages.
class SearchWorker_revamp {
 public:
 SearchWorker_revamp(Search_revamp* search, Node_revamp* worker_root, const SearchParams& params)
    : search_(search), params_(params), history_(search_->played_history_), worker_root_(worker_root) {}

  // Runs iterations while needed.
  void RunBlocking();
  void RunBlocking2();

 private:
  void AddNodeToComputation(Node_revamp* node);
  void AddNodeToComputation2();
  void retrieveNNResult(Node_revamp* node, int batchidx);
  void recalcPropagatedQ(Node_revamp* node);
  void computeWeights(Node_revamp* node, int depth);
  void computeWeights2(Node_revamp* node);
  int pickNodesToExtend(Node_revamp* current_node, int noof_nodes, int depth);
  std::vector<float> q_to_prob(std::vector<float> Q, int d, float multiplier, float max_focus);

  Search_revamp* const search_;
  std::vector<Node_revamp *> minibatch_;

  std::vector<float> weights_;
  std::vector<float> pvals_;
  std::vector<Node_revamp *> nodestack_;

  const SearchParams& params_;

  std::unique_ptr<CachingComputation> computation_;
  std::unique_ptr<NetworkComputation> computation2_;
  // History is reset and extended by PickNodeToExtend().
  PositionHistory history_;

  Node_revamp* worker_root_;
};


}  // namespace lczero
