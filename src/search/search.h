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
#include "search/params.h"
#include "search/node.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "syzygy/syzygy.h"
#include "utils/logging.h"
#include "utils/mutex.h"
#include "utils/optional.h"

namespace lczero {

struct SearchLimits {
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

class SearchCommon {
 public:
  SearchCommon(const NodeTreeCommon& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase *syzygy_tb);

  virtual ~SearchCommon() {}

  // Starts worker threads and returns immediately.
  virtual void StartThreads(size_t how_many) = 0;

  // Starts search with k threads and wait until it finishes.
  virtual void RunBlocking(size_t threads) = 0;

  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  virtual void Stop() = 0;
  // Stops search, but does not return bestmove. The function is not blocking.
  virtual void Abort() = 0;
  // Blocks until all worker thread finish.
  virtual void Wait() = 0;
  // Returns whether search is active. Workers check that to see whether another
  // search iteration is needed.
  virtual bool IsSearchActive() const = 0;

  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }

 protected:
  NNCache* cache_;
  SyzygyTablebase* syzygy_tb_;
  // Fixed positions which happened before the search.
  const PositionHistory& played_history_;

  Network* const network_;
  const SearchLimits limits_;
  const std::chrono::steady_clock::time_point start_time_;
  std::atomic<int> tb_hits_{0};

  BestMoveInfo::Callback best_move_callback_;
  ThinkingInfo::Callback info_callback_;
  const SearchParams params_;
};

}  // namespace lczero
