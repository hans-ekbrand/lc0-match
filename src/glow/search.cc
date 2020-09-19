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

#include "neural/encoder.h"

namespace lczero {


//////////////////////////////////////////////////////////////////////////////
// Search
//////////////////////////////////////////////////////////////////////////////

SearchGlow::SearchGlow(const NodeTreeGlow& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb,
               bool ponder)
    : SearchCommon(tree, network,
                   best_move_callback,
                   info_callback, limits,
                   options, cache,
                   syzygy_tb),
      root_node_(tree.GetCurrentHead()),
      initial_visits_(root_node_->GetN()),
      ponder_(ponder)
    {
			set_strategy_parameters(&params_);
		}

int64_t SearchGlow::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t SearchGlow::GetTimeToDeadline() const {
  if (!limits_.search_deadline) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             *limits_.search_deadline - std::chrono::steady_clock::now())
      .count();
}


void SearchGlow::StartThreads(size_t how_many) {
	threads_list_mutex_.lock();
	for (int i = threads_.size(); i < (int)how_many; i++) {
		n_thread_active_++;
		threads_.emplace_back([this, i]()
			{
				SearchWorkerGlow worker(this);
				worker.ThreadLoop(i);
			}
		);
	}
	threads_list_mutex_.unlock();
}

void SearchGlow::RunBlocking(size_t threads) {
	StartThreads(threads);
	Wait();
}

void SearchGlow::Stop() {
	ponder_lock_.lock();
	if (ponder_) {
		if (IsSearchActive()) {
			// If pondering is on, then turn if off to get bestmove
			ponder_ = false;
			not_stop_searching_ = false;
		} else {
		  // This makes us return a move for the opponent if we - Stop() - was called by PonderHit() in engine.cc
		  // To take care of this case, which happens when ponder is used together with node limits, we need a switch
		  // which tells us whether or not to report bestmove. Or PonderHit() should just not call us?
			reportBestMove();
		}
	} else {
		not_stop_searching_ = false;
	}
	ponder_lock_.unlock();
}

void SearchGlow::Abort() {
	abort_ = true;
	not_stop_searching_ = false;
}

bool SearchGlow::IsSearchActive() const {
	//threads_list_mutex_.lock();
	bool active = n_thread_active_ > 0;
	//threads_list_mutex_.unlock();
	return active;
}

NodeGlow *SearchGlow::indexOfHighestQEdge(NodeGlow* node, bool black_to_move, bool filter_out_unsafe_moves) {  
  // This function must only be called in the move selection process, by SendUciInfo() and reportBestMove()
  // black_to_move is only needed for reporting when we censored an uncertain move
  float highestq = -2.0;
  NodeGlow *bestidx = nullptr;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    float q = i->GetQ();
    assert((q >= -1) && (q >= 1));
    if (q > highestq) {
      highestq = q;
      bestidx = i;
    }
  }
  
  if(!filter_out_unsafe_moves){
    return bestidx;
  }

  // If bestidx is terminal, then just play it. Otherwise, play the move with the highest expected value.
  if(bestidx != nullptr && bestidx->IsTerminal()){
    LOGFILE << "Bestmove is terminal ";
    return bestidx;
  }

  // Put a rather strong prior on q (rescaled to [0,1]), so that a move candidate really have quite some visits that backup the claim that it is good.
  // Let's use the square root of the total number of visits as a prior, so alpha=1, beta=sqrt(N).
  // Or, make it pow(n, 0.3).
  // What we want to avoid is situations where the best q is, say 0.15 with 100 visits and next best q has 0.14 and 900 visits (of a total of 3000 visits).
  // In this situation, better go with 0.14.
  // The observational data is 100 visits, mean = 0.15, which we could express as: 100 = alpha + beta - 2; 0.15 = alpha / (alpha + beta);
  // that would give alpha = 15; beta = 85. Now add the prior
  // alpha = 16, beta = sqrt(3000) + 85 = 139.9
  // E = 16 / (16 + 139.9) = 0.10
  // For the child with 900 visits, we get alpha = 126, beta = 774, add the prior and we have alpha = 127, beta = 913
  // E = 127 / (127 + 913) = 0.12

  float beta_prior = pow(node->GetN(), 0.3);
  float alpha_prior = 1.0f;
  float highest_E = -1.0f;
  NodeGlow *really_bestidx = nullptr;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    float winrate = (i->GetQ() + 1) * 0.5;
    int visits = i->GetN();
    float alpha = winrate * visits + alpha_prior;
    float beta = visits - alpha + beta_prior;
    float E = alpha / (alpha + beta);
    if(E > highest_E){
      highest_E = E;
      really_bestidx = i;
    }
  }
    
  if(really_bestidx != bestidx){
    LOGFILE << "VETO against the uncertain move " << node->GetEdges()[bestidx->GetIndex()].GetMove(black_to_move).as_string() << " with only " << bestidx->GetN() << " visits and q = " << bestidx->GetQ();
    LOGFILE << "Best expected value after applying the prior as move " << node->GetEdges()[really_bestidx->GetIndex()].GetMove(black_to_move).as_string() << " with " << really_bestidx->GetN() << " visits and q = " << really_bestidx->GetQ();
    LOGFILE << "Beta prior applied for move selection: " << beta_prior << " based on " << node->GetN() << " visits at root";
  }
  return really_bestidx;
}

void SearchGlow::Wait() {
	threads_list_mutex_.lock();
	while (!threads_.empty()) {
		threads_.back().join();
		threads_.pop_back();
	}
	threads_list_mutex_.unlock();
}


SearchGlow::~SearchGlow() {
	Abort();
	Wait();
}

void SearchGlow::SendUciInfo() {

  auto score_type = params_.GetScoreType();

  ThinkingInfo common_info;
  if (root_node_->GetN() > initial_visits_)
    common_info.depth = cum_depth_ / (root_node_->GetN() - initial_visits_);
  common_info.seldepth = full_tree_depth_;
  common_info.time = GetTimeSinceStart();
  // common_info.nodes = root_node_->GetN();
  common_info.hashfull =
      cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  common_info.nps =
      common_info.time ? ((root_node_->GetN() - initial_visits_) * 1000 / common_info.time) : 0;
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  std::vector<ThinkingInfo> uci_infos;

  int multipv = 0;

  float prevq = 2.0;
  NodeGlow *previdx = nullptr;
  for (;;) {
    float bestq = -2.0;
    NodeGlow *bestidx = nullptr;
    for (NodeGlow *j = root_node_->GetFirstChild(); j != nullptr; j = j->GetNextSibling()) {
      float q = j->GetQ();
      if (q < prevq && q > bestq) {
        bestq = q;
        bestidx = j;
      } else if (q == prevq && q > bestq) {
	if (j == previdx) {
	  previdx = nullptr;
	} else if (previdx == nullptr) {
	  bestq = q;
	  bestidx = j;
	}
      }
    }
    if (bestidx == nullptr) break;
    prevq = bestq;
    previdx = bestidx;

    ++multipv;

    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();

    if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * bestq);
    } else if (score_type == "win_percentage") {
      uci_info.score = bestq * 5000 + 5000;
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * bestq);
    } else if (score_type == "Q") {
      uci_info.score = bestq * 10000;
    }

    if (params_.GetMultiPv() > 1) uci_info.multipv = multipv;
    bool flip = played_history_.IsBlackToMove();
    uci_info.pv.push_back(root_node_->GetEdges()[bestidx->GetIndex()].GetMove(flip));
    NodeGlow* n = bestidx;
    while (n && n->GetFirstChild() != nullptr) {
      flip = !flip;
      NodeGlow *bestidx = indexOfHighestQEdge(n, flip, false);
      uci_info.pv.push_back(n->GetEdges()[bestidx->GetIndex()].GetMove(flip));
      n = bestidx;
    }
    uci_info.nodes = bestidx->GetN();

    // LOGFILE << "nodes: " << bestidx->GetN() << " q: " << bestidx->GetQ() << " w: " << bestidx->GetW();
    
  }

  // reverse the order
  std::reverse(uci_infos.begin(), uci_infos.end());
  info_callback_(uci_infos);
}

void SearchGlow::checkLimitsAndMaybeTriggerStop() {
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

std::vector<std::string> SearchGlow::GetVerboseStats(NodeGlow* node, bool is_black_to_move) {

  std::vector<std::string> infos;
	for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
    std::ostringstream oss;
    oss << std::fixed;

    oss << std::left << std::setw(5)
        << node->GetEdges()[i->GetIndex()].GetMove(is_black_to_move).as_string();

    oss << " (" << std::setw(4) << node->GetEdges()[i->GetIndex()].GetMove(is_black_to_move).as_nn_index() << ")";

    oss << " N: " << std::right << std::setw(7) << i->GetN() << " (+"
        << std::setw(2) << i->GetN() << ") ";

    oss << "(P: " << std::setw(5) << std::setprecision(2) << node->GetEdges()[i->GetIndex()].GetP() * 100
        << "%) ";

    oss << "(Q: " << std::setw(8) << std::setprecision(5) << i->GetQ()
        << ") ";

    oss << "(U: " << std::setw(6) << std::setprecision(5) << i->GetQ()
        << ") ";

    oss << "(Q+U: " << std::setw(8) << std::setprecision(5)
        << i->GetQ() + i->GetQ() << ") ";

    oss << "(V: ";
    optional<float> v;
    if (i->IsTerminal()) {
      v = i->GetQ();
    } else {
      NNCacheLock nneval = GetCachedNNEval(i);
      if (nneval) v = -nneval->q;
    }
    if (v) {
      oss << std::setw(7) << std::setprecision(4) << *v;
    } else {
      oss << " -.----";
    }
    oss << ") ";

    if (i->IsTerminal()) oss << "(T) ";
    infos.emplace_back(oss.str());
  }
  std::reverse(infos.begin(), infos.end());
  return infos;
}

NNCacheLock SearchGlow::GetCachedNNEval(NodeGlow* node) const {
  if (!node) return {};

  std::vector<Move> moves;
  for (; node != root_node_; node = node->GetParent()) {
    moves.push_back(node->GetParent()->GetEdges()[node->GetIndex()].move_);
  }
  PositionHistory history(played_history_);
  for (auto iter = moves.rbegin(), end = moves.rend(); iter != end; ++iter) {
    history.Append(*iter);
  }
  auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

void SearchGlow::SendMovesStats() {
  const bool is_black_to_move = played_history_.IsBlackToMove();
  auto move_stats = SearchGlow::GetVerboseStats(root_node_, is_black_to_move);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    info_callback_(infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
}

void SearchGlow::reportBestMove() {
  NodeGlow *bestidx = indexOfHighestQEdge(root_node_, played_history_.IsBlackToMove(), true);
	Move best_move = root_node_->GetEdges()[bestidx->GetIndex()].GetMove(played_history_.IsBlackToMove());
	NodeGlow *ponderidx = indexOfHighestQEdge(bestidx, played_history_.IsBlackToMove(), false); // harmless to report a bad pondermove.
	// If the move we make is terminal, then there is nothing to ponder about.
	// Also, if the bestmove doesn't have any children, then don't report a ponder move.
	if(!bestidx->IsTerminal() && ponderidx != nullptr){
	  Move ponder_move = bestidx->GetEdges()[ponderidx->GetIndex()].GetMove(!played_history_.IsBlackToMove());
	  best_move_callback_({best_move, ponder_move});
	} else {
 	  LOGFILE << "in reportBestMove, Do not set a pondermove since the move we played was terminal.";	
	  best_move_callback_(best_move);
 	  LOGFILE << "in reportBestMove, Survived callback";
	}
}

void SearchGlow::ExtendNode(PositionHistory* history, NodeGlow* node) {
	// We don't need the mutex because other threads will see that N=0 and
	// N-in-flight=1 and will not touch this node.
	const auto& board = history->Last().GetBoard();
	auto legal_moves = board.GenerateLegalMoves();

	// Check whether it's a draw/lose by position. Importantly, we must check
	// these before doing the by-rule checks below.
	if (legal_moves.empty()) {
		// Could be a checkmate or a stalemate
		if (board.IsUnderCheck()) {
			node->MakeTerminal(GameResult::WHITE_WON);
		} else {
			node->MakeTerminal(GameResult::DRAW);
		}
		return;
	}

	// We can shortcircuit these draws-by-rule only if they aren't root;
	// if they are root, then thinking about them is the point.
	if (node != root_node_) {
		if (!board.HasMatingMaterial()) {
			node->MakeTerminal(GameResult::DRAW);
			return;
		}

		if (history->Last().GetNoCaptureNoPawnPly() >= 100) {
			node->MakeTerminal(GameResult::DRAW);
			return;
		}

		if (history->Last().GetRepetitions() >= 2) {
			node->MakeTerminal(GameResult::DRAW);
			return;
		}

		// Neither by-position or by-rule termination, but maybe it's a TB position.
		if (syzygy_tb_ && board.castlings().no_legal_castle() &&
				history->Last().GetNoCaptureNoPawnPly() == 0 &&
				(board.ours() | board.theirs()).count() <=
						syzygy_tb_->max_cardinality()) {
			ProbeState state;
			WDLScore wdl = syzygy_tb_->probe_wdl(history->Last(), &state);
			// Only fail state means the WDL is wrong, probe_wdl may produce correct
			// result with a stat other than OK.
			if (state != FAIL) {
				// If the colors seem backwards, check the checkmate check above.
				if (wdl == WDL_WIN) {
					node->MakeTerminal(GameResult::BLACK_WON);
				} else if (wdl == WDL_LOSS) {
					node->MakeTerminal(GameResult::WHITE_WON);
				} else {  // Cursed wins and blessed losses count as draws.
					node->MakeTerminal(GameResult::DRAW);
				}
				tb_hits_++;
				return;
			}
		}
	}

	// Add legal moves as edges of this node.
	node->CreateEdges(legal_moves);
}



}  // namespace lczero
