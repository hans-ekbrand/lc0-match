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

int const N_HELPER_THREADS_PRE = 5;
int const N_HELPER_THREADS_POST = 5;

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
  common_info.hashfull =
      cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  common_info.nps =
      common_info.time ? ((root_node_->GetN() - initial_visits_) * 1000 / common_info.time) : 0;
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

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
      NNCacheLock nneval = GetCachedNNEval(node->GetEdges()[i].GetChild());
      if (nneval) v = -nneval->q;
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

NNCacheLock Search_revamp::GetCachedNNEval(Node_revamp* node) const {
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

void Search_revamp::SendMovesStats() {
  const bool is_black_to_move = played_history_.IsBlackToMove();
  auto move_stats = Search_revamp::GetVerboseStats(root_node_, is_black_to_move);

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

void Search_revamp::reportBestMove() {
	int bestidx = indexOfHighestQEdge(root_node_);
	Move best_move = root_node_->GetEdges()[bestidx].GetMove(played_history_.IsBlackToMove());
	int ponderidx = indexOfHighestQEdge(root_node_->GetEdges()[bestidx].GetChild());
	// If the move we make is terminal, then there is nothing to ponder about.
	// Also, if the bestmove doesn't have any children, then don't report a ponder move.
	if(!root_node_->GetEdges()[bestidx].GetChild()->IsTerminal() &&
	   ponderidx != -1){
		Move ponder_move = root_node_->GetEdges()[bestidx].GetChild()->GetEdges()[ponderidx].GetMove(!played_history_.IsBlackToMove());
		best_move_callback_({best_move, ponder_move});
	} else {
		best_move_callback_(best_move);
	}
}

void Search_revamp::ExtendNode(PositionHistory* history, Node_revamp* node) {
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
				(board.ours() + board.theirs()).count() <=
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


//////////////////////////////////////////////////////////////////////////////
// Distribution
//////////////////////////////////////////////////////////////////////////////

float const Q_CONCENTRATION = 36.0;
int const MEMORY_FACTOR_THRESHOLD = 30;
float const MEMORY_FACTOR_INITIAL = 0.95;
float const MEMORY_FACTOR_FINAL = 0.0;


float compute_d_fun_1_par(float half_life) {
  return log(0.5) / half_life;
}

// par should be negative
inline float d_fun_1(float par, float x) {
  return exp(par * x);
}

// par1 should be positive and par2 should be negative
float compute_d_fun_2_par1_from_par2(float par2, float half_life) {
  return (pow(0.5, 1.0/par2) - 1.0) / half_life;
}

// par1 should be positive and par2 should be negative
inline float d_fun_2(float par1, float par2, float x) {
  return pow(1.0 + par1 * x, par2);
}

float const half_life_q_to_w = compute_d_fun_1_par(-Q_CONCENTRATION);

float const d_fun_1_par_q_to_w = compute_d_fun_1_par(half_life_q_to_w);

float const D_FUN_2_PAR2_Q_TO_W = -2.0;
float const d_fun_2_par1_q_to_w = compute_d_fun_2_par1_from_par2(D_FUN_2_PAR2_Q_TO_W, half_life_q_to_w);

inline float d_fun_1_q_to_w(float highest_q, float q) {
  return d_fun_1(d_fun_1_par_q_to_w, highest_q - q);
}

inline float d_fun_2_q_to_w(float highest_q, float q) {
  return d_fun_2(d_fun_2_par1_q_to_w, D_FUN_2_PAR2_Q_TO_W, highest_q - q);
}


void SearchWorker_revamp::recalcNode(Node_revamp *node) {
  int old_n = node->GetN();
  int n = 1;
  float highest_q = 0.0;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    n += node->GetEdges()[i].GetChild()->GetN();
    float q = node->GetEdges()[i].GetChild()->GetQ();
    if (q > highest_q) highest_q = q;
  }
  node->SetN(n);

  float memory_factor = pow(n < MEMORY_FACTOR_THRESHOLD ? MEMORY_FACTOR_INITIAL : MEMORY_FACTOR_FINAL, n - old_n);
  float one_minus_memory_factor = 1.0 - memory_factor;

  float w_total = 0.0;
  float p_total = 0.0;
  float q_total = 0.0;
  float max_max_w = 0.0;
  int16_t max_max_w_idx = -1;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    float q = node->GetEdges()[i].GetChild()->GetQ();
    float w = memory_factor * node->GetEdges()[i].GetChild()->GetW() + one_minus_memory_factor * d_fun_1_q_to_w(highest_q, q);
    node->GetEdges()[i].GetChild()->SetW(w);
    w_total += w;
    q_total -= w * q;
    p_total += node->GetEdges()[i].GetP();
    float max_w = w * node->GetEdges()[i].GetChild()->GetMaxW();
    if (max_w > max_max_w) {
      max_max_w = max_w;
      max_max_w_idx = i;
    }
  }

  node->SetQ(p_total * q_total / w_total + (1.0 - p_total) * node->GetOrigQ());
  max_max_w *= p_total / w_total;

  int nidx = node->GetNextUnexpandedEdge();
  if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
    float max_w = node->GetEdges()[nidx].GetP();
    if (max_w > max_max_w) {
      max_max_w = max_w;
      max_max_w_idx = -1;
    }
  }
  node->SetMaxW(max_max_w);
  node->SetBestIdx(max_max_w_idx);
}


void SearchWorker_revamp::pickNodesToExtend() {
	Node_revamp* node;
	int best_idx;

	int nodes_visited = 0;

	for (int n = 0; n < new_nodes_amount_target_ && n < new_nodes_amount_limit_; n++) {
		node = root_node_;

		while (true) {
			nodes_visited++;
			best_idx = node->GetBestIdx();
			if (best_idx == -1) {
				int nidx = node->GetNextUnexpandedEdge();
				if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
					node->GetEdges()[nidx].CreateChild(node, nidx);
					new_nodes_[new_nodes_size_] = {node->GetEdges()[nidx].GetChild(), 0xFFFF, -1};
					new_nodes_size_++;
					node->SetNextUnexpandedEdge(nidx + 1);
					break;
				} else {  // no more child to add (before retrieved information about previous ones)
					return;
				}
			}
			node = node->GetEdges()[best_idx].GetChild();
		}

		int junction_mode = 0;
		uint16_t ccidx = (new_nodes_size_ - 1) | 0x8000;

		while (true) {
      float w_total = 0.0;
      float p_total = 0.0;
      float max_max_w = 0.0;
      int16_t max_max_w_idx = -1;
      for (int i = 0; i < node->GetNumChildren(); i++) {
        float w = node->GetEdges()[i].GetChild()->GetW();
        w_total += w;
        p_total += node->GetEdges()[i].GetP();
        float max_w = w * node->GetEdges()[i].GetChild()->GetMaxW();
        if (max_w > max_max_w) {
          max_max_w = max_w;
          max_max_w_idx = i;
        }
      }
      max_max_w *= p_total / w_total;
      int nidx = node->GetNextUnexpandedEdge();
      if (nidx < node->GetNumEdges() && nidx - node->GetNumChildren() < MAX_NEW_SIBLINGS) {
        float max_w = node->GetEdges()[nidx].GetP();
        if (max_w > max_max_w) {
          max_max_w = max_w;
          max_max_w_idx = -1;
        }
      }
      node->SetMaxW(max_max_w);
      node->SetBestIdx(max_max_w_idx);


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

void SearchWorker_revamp::buildJunctionRTree() {
	for (int i = new_nodes_size_ - 1; i >= 0; i--) {
		uint16_t junction = new_nodes_[i].junction;
		while (junction != 0xFFFF) {
			int cc = junctions_[junction].children_count++;
			if (cc > 0) break;
			junction = junctions_[junction].parent;
		}
	}

	for (int i = new_nodes_size_ - 1; i >= 0; i--) {
		Node_revamp* node = new_nodes_[i].node->GetParent();
		while (node->GetBranchingInFlight() != 0) {
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


int SearchWorker_revamp::AddNodeToComputation(Node_revamp* node, PositionHistory *history) {
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


int SearchWorker_revamp::extendTree(std::vector<Move> *movestack, PositionHistory *history) {
	int count = 0;

	int full_tree_depth = search_->full_tree_depth_;
	int cum_depth = 0;

	while (true) {
		new_nodes_list_lock_.lock();

		int i = new_nodes_list_shared_idx_;
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

		Node_revamp* newchild = new_nodes_[i].node;

		count++;

    newchild->SetW(newchild->GetParent()->GetEdges()[newchild->GetIndex()].GetP() / newchild->GetParent()->GetEdges()[0].GetP());

		int nappends = appendHistoryFromTo(movestack, history, root_node_, newchild);

		search_->ExtendNode(history, newchild);

		if (!newchild->IsTerminal()) {

			int idx = AddNodeToComputation(newchild, history);
			new_nodes_[i].batch_idx = idx;

		} else {  // is terminal
      new_nodes_amount_target_++;  // it's terminal so it shouldn't be counted towards the minibatch size
		}

		history->Trim(played_history_length_);
		//for (int j = 0; j <= nappends; j++) {
		//	history->Pop();
		//}

		if (nappends - 1 > full_tree_depth) full_tree_depth = nappends - 1;
		cum_depth += nappends - 1;

		}
	}

	if (full_tree_depth > search_->full_tree_depth_) search_->full_tree_depth_ = full_tree_depth;
	search_->cum_depth_ += cum_depth;

	return count;
}


void SearchWorker_revamp::retrieveNNResult(Node_revamp* node, int batchidx) {
  float q = -computation_->GetQVal(batchidx);
  if (q < -1.0 || q > 1.0) {
    std::cerr << "q = " << q << "\n";
    abort();
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


int SearchWorker_revamp::propagate() {
	int count = 0;

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
			Node_revamp* node = new_nodes_[j].node->GetParent();
			uint16_t juncidx = new_nodes_[j].junction;

			while (juncidx != 0xFFFF) {
				while (node != junctions_[juncidx].node) {
					recalcNode(node);
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
					recalcNode(node);
					count++;
					if (node == root_node_) break;
					node = node->GetParent();
				}
			}
		}
	}

  search_->count_propagate_node_visits_ += count;

	return count;
}



void SearchWorker_revamp::ThreadLoop(int thread_id) {

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

		helper_threads_mode_ = 1;
		//LOGFILE << "Allowing helper threads to help";
		for (int j = 0; j < N_HELPER_THREADS_PRE; j++) {
			helper_thread_locks[j]->unlock();
		}

		auto start_comp_time = std::chrono::steady_clock::now();
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

		auto stop_comp_time = std::chrono::steady_clock::now();
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
            << ", new_nodes_amount_target_: " << new_nodes_amount_target_
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
      search_->count_minibatch_size_ += minibatch_shared_idx_;
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
		for (int j = 0; j < N_HELPER_THREADS_POST; j++) {
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
		if (!ponder && !search_->abort_) {
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
	    for (int i = 0; i < root_node_->GetNumChildren(); i++) {
	      LOGFILE << std::fixed << std::setfill(' ') 
		      << (root_node_->GetEdges())[i].move_.as_string() << " "
		      << std::setw(10) << (root_node_->GetEdges())[i].GetP() << " "
		      << std::setw(10) << (root_node_->GetEdges())[i].GetChild()->GetN() << " "
		      << std::setw(10) << (float)(root_node_->GetEdges())[i].GetChild()->GetN() / (float)(root_node_->GetN() - 1) << " "
		// << std::setw(4) << (root_node_->GetEdges())[i].GetChild()->ComputeHeight() << " "
		      << std::setw(10) << (float)(root_node_->GetEdges())[i].GetChild()->GetQ() << " "
		      << std::setw(10) << root_node_->GetEdges()[i].GetChild()->GetW();
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
