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

#include "glow/node.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include "neural/encoder.h"
#include "neural/network.h"
#include "utils/exception.h"
#include "utils/hashcat.h"

namespace lczero {


/////////////////////////////////////////////////////////////////////////
// NodeGlow garbage collector
/////////////////////////////////////////////////////////////////////////

namespace {
// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollectorGlow {
 public:
  NodeGarbageCollectorGlow() : gc_thread_([this]() { Worker(); }) {}

  // Takes ownership of a subtree, to dispose it in a separate thread when
  // it has time.
  void AddToGcQueue(std::unique_ptr<NodeGlow> node) {
    if (!node) return;
    Mutex::Lock lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
  }

  ~NodeGarbageCollectorGlow() {
    // Flips stop flag and waits for a worker thread to stop.
    stop_ = true;
    gc_thread_.join();
  }

 private:
  void GarbageCollect() {
    while (!stop_) {
      // NodeGlow will be released in destructor when mutex is not locked.
      std::unique_ptr<NodeGlow> node_to_gc;
      {
        // Lock the mutex and move last subtree from subtrees_to_gc_ into
        // node_to_gc.
        Mutex::Lock lock(gc_mutex_);
        if (subtrees_to_gc_.empty()) return;
        node_to_gc = std::move(subtrees_to_gc_.back());
        subtrees_to_gc_.pop_back();
      }
    }
  }

  void Worker() {
    while (!stop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    };
  }

  mutable Mutex gc_mutex_;
  std::vector<std::unique_ptr<NodeGlow>> subtrees_to_gc_ GUARDED_BY(gc_mutex_);

  // When true, Worker() should stop and exit.
  volatile bool stop_ = false;
  std::thread gc_thread_;
};  // namespace

NodeGarbageCollectorGlow gNodeGc;
}  // namespace

/////////////////////////////////////////////////////////////////////////
// EdgeGlow
/////////////////////////////////////////////////////////////////////////

Move EdgeGlow::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Mirror();
  return m;
}


// Policy priors (P) are stored in a compressed 16-bit format.
//
// Source values are 32-bit floats:
// * bit 31 is sign (zero means positive)
// * bit 30 is sign of exponent (zero means nonpositive)
// * bits 29..23 are value bits of exponent
// * bits 22..0 are significand bits (plus a "virtual" always-on bit: s ∈ [1,2))
// The number is then sign * 2^exponent * significand, usually.
// See https://www.h-schmidt.net/FloatConverter/IEEE754.html for details.
//
// In compressed 16-bit value we store bits 27..12:
// * bit 31 is always off as values are always >= 0
// * bit 30 is always off as values are always < 2
// * bits 29..28 are only off for values < 4.6566e-10, assume they are always on
// * bits 11..0 are for higher precision, they are dropped leaving only 11 bits
//     of precision
//
// When converting to compressed format, bit 11 is added to in order to make it
// a rounding rather than truncation.
//
// Out of 65556 possible values, 2047 are outside of [0,1] interval (they are in
// interval (1,2)). This is fine because the values in [0,1] are skewed towards
// 0, which is also exactly how the components of policy tend to behave (since
// they add up to 1).

// If the two assumed-on exponent bits (3<<28) are in fact off, the input is
// rounded up to the smallest value with them on. We accomplish this by
// subtracting the two bits from the input and checking for a negative result
// (the subtraction works despite crossing from exponent to significand). This
// is combined with the round-to-nearest addition (1<<11) into one op.
void EdgeGlow::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float EdgeGlow::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string EdgeGlow::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.as_string() << " p_: " << p_ << " GetP: " << GetP();
  return oss.str();
}

/////////////////////////////////////////////////////////////////////////
// EdgeListGlow
/////////////////////////////////////////////////////////////////////////

EdgeListGlow::EdgeListGlow(MoveList moves)
    : edges_(std::make_unique<EdgeGlow[]>(moves.size())), size_(moves.size()) {
  auto* edge = edges_.get();
  for (auto move : moves) edge++->move_ = move;
}

/////////////////////////////////////////////////////////////////////////
// NodeGlow
/////////////////////////////////////////////////////////////////////////

NodeGlow* NodeGlow::CreateSingleChildNode(Move move) {
  assert(!edges_);
  assert(!child_); // error: ‘child_’ was not declared in this scope
  edges_ = EdgeListGlow({move});
	AddChild(std::make_unique<NodeGlow>(this, 0));
	return GetFirstChild();
}

void NodeGlow::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_); // error: ‘child_’ was not declared in this scope
  edges_ = EdgeListGlow(moves);
}

// assumes child_ to be nullptr
void NodeGlow::SortEdgesByPValue() {
  int n = edges_.size();
  for (int i = 0; i < n - 1; i++) {
    float best = edges_[i].GetP();
    int bestidx = i;
    for (int j = i + 1; j < n; j++) {
      if (edges_[j].GetP() > best) {
        best = edges_[j].GetP();
        bestidx = j;
      }
    }
    if (bestidx != i) {
      Move tmpmove = edges_[bestidx].move_;
      edges_[bestidx].move_ = edges_[i].move_;
      edges_[bestidx].SetP(edges_[i].GetP());
      edges_[i].move_ = tmpmove;
      edges_[i].SetP(best);
    }
  }
}

void NodeGlow::AddChild(std::unique_ptr<NodeGlow> node) {
	std::unique_ptr<NodeGlow> oldchild = std::move(child_);  // or maybe put it last in list
  child_ = std::move(node);
//  parent->noofchildren_++;  // update noofchildren when node is ready (has received nn values)
	child_->sibling_ = std::move(oldchild);
	noofchildren_++;
}



std::string NodeGlow::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << is_terminal_ << " This:" << this << " Parent:" << parent_
      << " Index:" << index_ << " Q:" << q_
      << " Edges:" << edges_.size();
  return oss.str();
}

void NodeGlow::MakeTerminal(GameResult result) {
  is_terminal_ = true;
  if (result == GameResult::DRAW) {
    SetOrigQ(0.0f);
  } else if (result == GameResult::WHITE_WON) {
    SetOrigQ(1.0f);
  } else if (result == GameResult::BLACK_WON) {
    SetOrigQ(-1.0f);
  }
}

void NodeGlow::ReleaseChildren() {
  gNodeGc.AddToGcQueue(std::move(child_));
}

void NodeGlow::ReleaseChildrenExceptOne(NodeGlow* node_to_save) {
	NodeGlow* cur = child_.get();
	NodeGlow* prev = nullptr;
	while (cur != nullptr && cur != node_to_save) {
		prev = cur;
		cur = cur->sibling_.get();
	}
	if (cur == nullptr) {
		gNodeGc.AddToGcQueue(std::move(child_));
	} else {
		gNodeGc.AddToGcQueue(std::move(cur->sibling_));
		if (prev != nullptr) {
			std::unique_ptr<NodeGlow> the_node = std::move(prev->sibling_);
			gNodeGc.AddToGcQueue(std::move(child_));
			child_ = std::move(the_node);
		}
	}
}





/////////////////////////////////////////////////////////////////////////
// NodeTreeGlow
/////////////////////////////////////////////////////////////////////////

void NodeTreeGlow::MakeMove(Move move) {
  if (HeadPosition().IsBlackToMove()) move.Mirror();

  NodeGlow* new_head = current_head_->child_.get();
	while (new_head != nullptr && current_head_->edges_[new_head->index_].move_ != move) {
		new_head = new_head->sibling_.get();
	}
	current_head_->ReleaseChildrenExceptOne(new_head);
  current_head_ =
      new_head ? new_head : current_head_->CreateSingleChildNode(move);
  history_.Append(move);
}

void NodeTreeGlow::TrimTreeAtHead() {
  // Send dependent nodes for GC instead of destroying them immediately.
  current_head_->ReleaseChildren();
  *current_head_ = NodeGlow(current_head_->GetParent(), current_head_->index_);
}

bool NodeTreeGlow::ResetToPosition(const std::string& starting_fen,
                               const std::vector<Move>& moves) {
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);
  if (gamebegin_node_ && history_.Starting().GetBoard() != starting_board) {
    // Completely different position.
    DeallocateTree();
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<NodeGlow>(nullptr, 0);
  }

  history_.Reset(starting_board, no_capture_ply,
                 full_moves * 2 - (starting_board.flipped() ? 1 : 2));

  NodeGlow* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const auto& move : moves) {
    MakeMove(move);
    if (old_head == current_head_) seen_old_head = true;
  }

  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  // Also, if the current_head_ is terminal, reset that as well to allow forced
  // analysis of WDL hits, or possibly 3 fold or 50 move "draws", etc.
  if (!seen_old_head || current_head_->IsTerminal()) TrimTreeAtHead();

  return seen_old_head;
}

void NodeTreeGlow::DeallocateTree() {
  // Same as gamebegin_node_.reset(), but actual deallocation will happen in
  // GC thread.
  gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}


}  // namespace lczero
