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

#include <iostream>
#include <fstream>
#include <math.h>
#include <cassert>  // assert() used for debugging during development
#include <iomanip>

#include "neural/encoder.h"

namespace lczero {

namespace {

// Alternatives:

int const Q_TO_PROB_MODE = 1;
  // 1: e^(k * q)
  // 2: 1 / (1 + k (maxq - q))^2

int const MAX_NEW_SIBLINGS = 10000;
  // The maximum number of new siblings. If 1, then it's like old MULTIPLE_NEW_SIBLINGS = false, if >= maximum_number_of_legal_moves it's like MULTIPLE_NEW_SIBLINGS = true
const int kUciInfoMinimumFrequencyMs = 5000;

int const N_HELPER_THREADS_PRE = 5;
int const N_HELPER_THREADS_POST = 5;

bool const LOG_RUNNING_INFO = false;

float const policy_exponent_minimum = 20; // linear decrease with number of subnodes until this value is reached.

}  // namespace


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
    {}

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

namespace {

  int indexOfHighestQEdge(NodeGlow* node, bool black_to_move, bool filter_uncertain_moves) {
    float highestq = -2.0;
    int bestidx = -1;
    for (int i = 0; i < node->GetNumChildren(); i++) {
      float q = node->GetEdges()[i].GetChild()->GetQ();
      if(q < -1 || q > 1){
	LOGFILE << "Warning abs(Q) is above 1, q=" << q;
      }
      if (q > highestq) {
	highestq = q;
	bestidx = i;
      }
    }
    // return bestidx;
    // This is mostly relevant for games played with very low nodecounts.
    // Veto moves with too high uncertainty in Q, by requiring at least 3 * log(n) visits if number of subnodes is above n, and the suggested move is not a terminal node. TODO use some lookup table for log here
    unsigned int threshold = ceil(3 * log(node->GetN()));
    if (! filter_uncertain_moves ||
    	node->GetN() < 1000 ||
    	node->GetEdges()[bestidx].GetChild()->GetN() >= threshold ||
        node->GetEdges()[bestidx].GetChild()->IsTerminal())
    	 {
    	   return bestidx;
    }

    // Search until an acceptable move is found. Should be a rare event to even end up here, so no point in optimising the code below.
    std::vector<int> bad_moves(1);
    bad_moves[0] = bestidx;
    while(true){
      LOGFILE << "VETO against the uncertain move " << node->GetEdges()[bestidx].GetMove(black_to_move).as_string() << " with only " << node->GetEdges()[bestidx].GetChild()->GetN() << " visits. Not acceptable.";
      highestq = -2.0;
      for (int i = 0; i < node->GetNumChildren(); i++) {
    	if ( std::find(bad_moves.begin(), bad_moves.end(), i) == bad_moves.end() ){ // no match
    	  float q = node->GetEdges()[i].GetChild()->GetQ();
    	  if (q > highestq) {
    	    highestq = q;
    	    bestidx = i;
    	  }
    	}
      }
      if (node->GetEdges()[bestidx].GetChild()->GetN() >= threshold ||
    	  node->GetEdges()[bestidx].GetChild()->IsTerminal()) {
    	return bestidx;
      } else {
    	// add bestidx to the list of unacceptable moves
    	LOGFILE << "So many bad moves. Sad.";
    	bad_moves.push_back(bestidx);
    	LOGFILE << "Storing succeeded.";	
      }
    }
  }
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
    NodeGlow* n = root_node_->GetEdges()[bestidx].GetChild();
    while (n && n->GetNumChildren() > 0) {
      flip = !flip;
      int bestidx = indexOfHighestQEdge(n, played_history_.IsBlackToMove(), true); // Filter out uncertain moves from uci-info.
      uci_info.pv.push_back(n->GetEdges()[bestidx].GetMove(flip));
      n = n->GetEdges()[bestidx].GetChild();
    }
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
	int bestidx = indexOfHighestQEdge(root_node_, played_history_.IsBlackToMove(), true);
	Move best_move = root_node_->GetEdges()[bestidx].GetMove(played_history_.IsBlackToMove());
	int ponderidx = indexOfHighestQEdge(root_node_->GetEdges()[bestidx].GetChild(), played_history_.IsBlackToMove(), false); // harmless to report a bad pondermove.
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


//////////////////////////////////////////////////////////////////////////////
// Distribution
//////////////////////////////////////////////////////////////////////////////

	// Parameters we use
	// FpuValue: policy_weight_exponent 0.59
  	// Temperature: q_concentration 36.2
        // MaxCollisionVisits: number of sub nodes before exploration encouragement kicks in. 900 is reasonable, but for long time control games perhaps better leave it at 1.
  	// TemperatureVisitOffset: coefficient by which q_concentration is reduced per subnode. 0.0000082
	// TemperatureWinpctCutoff: factor to boost cpuct * policy = explore moves with high policy but low q. reasonable value: 3-5 (when TemperatureVisitOffset is non-zero lower, say 0.5)
	// Cpuct: factor to boost cpuct = explore moves regardless of their policy reasonable value: 0.003

  inline float q_to_prob(const float q, const float max_q, const float q_concentration, int n, int parent_n) {
  switch (Q_TO_PROB_MODE) {
  case 1: {
    // double my_q_concentration_ = 35.2;
    // double my_q_concentration_ = 40;
    // When a parent is well explored, we need to reward exploration instead of (more) exploitation.
    // A0 and Lc0 does this by increase policy on children with relatively few visits.
    // Let's try to do that instead by decreasing q_concentration as parent N increases.
    // At higher visits counts, our policy term has no influence anymore.
    // I've modelled a function after the dynamic cpuct invented by DeepMind, so that our function decreases by half at the same number parent nodes as the the dynamic cpuct is doubled (for zero visit at the child). We reward exploration regardless of number of child visits, which might not be as effective as their strategy, but let's give it a go.
    // return exp(q_concentration * (0.246 + (1 - 0.246) / pow((1 + parent_n / 30000), 0.795)) * (q - abs(max_q)/2)); // reduce the overflow risk.
    // if(parent_n > 100000){
    //   // Reduce q_concentration to 35.3 by 1E6 and 34.8 by 3E6.
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n)/2.5 - 4.6);
    //   // Reduce q_concentration to 34.7 by 1E6 and 33.9 by 3E6.      
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n)/1.5 - 7.67);
    //   // Reduce q_concentration to 34.3 by 1E6 and 33.4 by 3E6.            
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n)/1.2 - 9.59);            
    //   // Reduce q_concentration to 33.9 by 1E6 and 32.8 by 3E6.            
    //   // float dynamic_q_concentration = q_concentration - (log(parent_n) - 11.51);
    //   // cyclic, mean = q_concentration - amplitude
    //   float amplitude = 1.1;
    //   float dynamic_q_concentration = (1 + cos(3.141592 * n / 50000)) * amplitude + q_concentration - 2 * amplitude;
    //   return exp(dynamic_q_concentration * (q - abs(max_q)/2)); // reduce the overflow risk.
    // } else {
      return exp(q_concentration * (q - abs(max_q)/2)); // reduce the overflow risk. However, with the default q_concentration 36.2, overflow isn't possible since exp(36.2 * 1) is less than max float. TODO restrict the parameter so that it cannot overflow and remove this division.
    // }
  };
  case 2: {
    float x = 1.0 + 20.0 * (max_q - q);
    return pow(x, -2.0);
  };
  };
}


  float SearchWorkerGlow::computeChildWeights(NodeGlow* node, bool evaluation_weights) {
  int n = node->GetNumChildren();
  float maxq = -2.0;
  for (int i = 0; i < n; i++) {
    float q = node->GetEdges()[i].GetChild()->GetQ();
    if (q > maxq) maxq = q;
  }

  float sum_of_P_of_expanded_nodes = 0.0;
  float sum_of_w_of_expanded_nodes = 0.0;
  float sum_of_weighted_p_and_q = 0.0;
  for (int i = 0; i < n; i++) {
    float w = q_to_prob(node->GetEdges()[i].GetChild()->GetQ(), maxq, search_->params_.GetTemperature(), node->GetEdges()[i].GetChild()->GetN(), node->GetN());
    node->GetEdges()[i].GetChild()->SetW(w);
    sum_of_w_of_expanded_nodes += w;
    sum_of_P_of_expanded_nodes += node->GetEdges()[i].GetP();
  }
  float normalise_to_sum_of_p = sum_of_P_of_expanded_nodes / sum_of_w_of_expanded_nodes; // Avoid division in the loop, multiplication should be faster.
  std::vector<float> weighted_p_and_q(n);
  float relative_weight_of_p = 0;
  float relative_weight_of_q = 0;
  for (int i = 0; i < n; i++) {
    node->GetEdges()[i].GetChild()->SetW(node->GetEdges()[i].GetChild()->GetW() * normalise_to_sum_of_p); // Normalise sum of Q to sum of P
    relative_weight_of_p = pow(node->GetEdges()[i].GetChild()->GetN(), search_->params_.GetFpuValue(false)) / (0.05 + node->GetEdges()[i].GetChild()->GetN()); // 0.05 is here to make Q have some influence after 1 visit.
    float relative_weight_of_q = 1 - relative_weight_of_p;
    weighted_p_and_q[i] = relative_weight_of_q * node->GetEdges()[i].GetChild()->GetW() + relative_weight_of_p * node->GetEdges()[i].GetP();  // Weight for evaluation
    sum_of_weighted_p_and_q += weighted_p_and_q[i];
  }

  // Normalise the weighted sum Q + P to sum of P
  float normalise_weighted_sum = sum_of_P_of_expanded_nodes / sum_of_weighted_p_and_q;
  for (int i = 0; i < n; i++) {
    node->GetEdges()[i].GetChild()->SetW(weighted_p_and_q[i] * normalise_weighted_sum);
  }

  // If evalution_weights is requested, then we are done now.
  if(evaluation_weights){
    return(sum_of_P_of_expanded_nodes);
  }
    
  // For now, just copy and paste the above and redo it using different formulas
  // There are 3 independent mechanisms to encourage exploration
  // New Q based on decreasing q_conc: TemperatureVisitOffset --temp-visit-offset
  // Boost policy: TemperatureWinpctCutoff --temp-value-cutoff
  // Boost all moves with few visits: CPuct --cpuct
  // ./lc0 -w /home/hans/32603 --cpuct=0.003 --temp-visit-offset=0.0000082 --temp-value-cutoff=3 --fpu-value=0.59 --temperature=36.2 --verbose-move-stats --logfile=\<stderr\> --policy-softmax-temp=1.0 --max-collision-visits=1 
  
  sum_of_w_of_expanded_nodes = 0.0;
  float q_conc = search_->params_.GetTemperature() - search_->params_.GetTemperatureVisitOffset() * node->GetN();
  if(q_conc < policy_exponent_minimum){
    q_conc = policy_exponent_minimum;
  }
  for (int i = 0; i < n; i++) {
    float w = q_to_prob(node->GetEdges()[i].GetChild()->GetQ(), maxq, q_conc, node->GetEdges()[i].GetChild()->GetN(), node->GetN());
    node->GetEdges()[i].GetChild()->SetW(w);
    sum_of_w_of_expanded_nodes += w;
  }
  normalise_to_sum_of_p = sum_of_P_of_expanded_nodes / sum_of_w_of_expanded_nodes; // Avoid division in the loop, multiplication should be faster.
  for (int i = 0; i < n; i++) {
    float cpuct = log((node->GetN() + search_->params_.GetCpuctBase())/search_->params_.GetCpuctBase()) * sqrt(log(node->GetN())/(1+node->GetEdges()[i].GetChild()->GetN()));
    float exploration_term = search_->params_.GetTemperatureWinpctCutoff() * cpuct * node->GetEdges()[i].GetP() + search_->params_.GetCpuct() * cpuct;
    // TODO reuse this
    relative_weight_of_p = pow(node->GetEdges()[i].GetChild()->GetN(), search_->params_.GetFpuValue(false)) / (0.05 + node->GetEdges()[i].GetChild()->GetN()); // 0.05 is here to make Q have some influence after 1 visit.
    // relative_weight_of_p = capped_cpuct > relative_weight_of_p ? capped_cpuct : relative_weight_of_p; // If capped cpuct is greater than relative_weight_of_p, then use it instead.
    relative_weight_of_q = 1 - relative_weight_of_p;
    weighted_p_and_q[i] = relative_weight_of_q * node->GetEdges()[i].GetChild()->GetW() * normalise_to_sum_of_p + relative_weight_of_p * node->GetEdges()[i].GetP() + exploration_term; // Weight for exploration
    sum_of_weighted_p_and_q += weighted_p_and_q[i];
  }
  float final_normalisation = sum_of_P_of_expanded_nodes / sum_of_weighted_p_and_q;
  for (int i = 0; i < n; i++){
    node->GetEdges()[i].GetChild()->SetW(weighted_p_and_q[i] * final_normalisation);
  }
  return(sum_of_P_of_expanded_nodes);
}


void SearchWorkerGlow::pickNodesToExtend() {
	NodeGlow* node;
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
		NodeGlow* node = new_nodes_[i].node->GetParent();
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

		NodeGlow* newchild = new_nodes_[i].node;
		//int idx = new_nodes_[i].idx;

		count++;

		int nappends = appendHistoryFromTo(movestack, history, root_node_, newchild);
		//NodeGlow* newchild = node->GetEdges()[idx].GetChild();

		//history->Append(node->GetEdges()[idx].move_);

		search_->ExtendNode(history, newchild);

		if (!newchild->IsTerminal()) {

			int idx = AddNodeToComputation(newchild, history);
			new_nodes_[i].batch_idx = idx;
			//minibatch_.push_back({newchild, (uint16_t)i});
			//LOGFILE << "minibatch add: " << new_nodes_[i].junction;

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


void SearchWorkerGlow::recalcPropagatedQ(NodeGlow* node) {
  int n = 1;
  for (int i = 0; i < node->GetNumChildren(); i++) {
    n += node->GetEdges()[i].GetChild()->GetN();
  }
  node->SetN(n);

  float total_children_weight = computeChildWeights(node, true);

  // Average Q START
  float q = (1.0 - total_children_weight) * node->GetOrigQ();
  for (int i = 0; i < node->GetNumChildren(); i++) {
    q -= node->GetEdges()[i].GetChild()->GetW() * node->GetEdges()[i].GetChild()->GetQ();
  }
  node->SetQ(q);
  // Average Q STOP

  // When the node get more than MaxCollisionVisits subnodes, then recalc w to encourage exploration
  if(node->GetN() > (uint32_t)search_->params_.GetMaxCollisionVisitsId()){
      total_children_weight = computeChildWeights(node, false);
  }
  
  int first_non_created_child_idx = node->GetNumChildren();
  while (first_non_created_child_idx < node->GetNumEdges() && node->GetEdges()[first_non_created_child_idx].GetChild() != nullptr) {
    first_non_created_child_idx++;
  }

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
			NodeGlow* node = new_nodes_[j].node->GetParent();
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
