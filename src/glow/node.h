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

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/position.h"
#include "neural/writer.h"
#include "utils/mutex.h"
#include "search/node.h"

namespace lczero {


class NodeGlow;
class EdgeGlow {
 public:
  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  NodeGlow* GetChild() { return child_.get(); }
  NodeGlow* CreateChild(NodeGlow* parent, uint16_t index);

  // Debug information about the edge.
  std::string DebugString() const;

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Pointer to child of this edge. nullptr for no node.
  std::unique_ptr<NodeGlow> child_ = nullptr;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;

  friend class EdgeListGlow;
  friend class NodeGlow;
  friend class NodeTreeGlow;
	friend class SearchGlow;
	friend class SearchWorkerGlow;
};


// Array of Edges.
class EdgeListGlow {
 public:
  EdgeListGlow() {}
  EdgeListGlow(MoveList moves);
  EdgeGlow* get() const { return edges_.get(); }
  EdgeGlow& operator[](size_t idx) const { return edges_[idx]; }
  operator bool() const { return static_cast<bool>(edges_); }
  uint16_t size() const { return size_; }

 private:
  std::unique_ptr<EdgeGlow[]> edges_;
  uint16_t size_ = 0;
};


class NodeGlow {
 public:
  // Takes pointer to a parent node and own index in a parent.
 NodeGlow(NodeGlow* parent, uint16_t index) : parent_(parent), index_(index) {}

  // Allocates a new edge and a new node. The node has to be no edges before
  // that.
  NodeGlow* CreateSingleChildNode(Move m);

  // Creates edges from a movelist. There has to be no edges before that.
  void CreateEdges(const MoveList& moves);
  
  void SortEdgesByPValue();

  // Gets parent node.
  NodeGlow* GetParent() const { return parent_; }

  // Returns node eval, i.e. average subtree V for non-terminal node and -1/0/1
  // for terminal nodes.
  float GetQ() const { return q_; }
  void SetQ(float q) { q_ = q; }
  float GetQInacc() const { return q_inacc_; }
  void SetQInacc(float q_inacc) { q_inacc_ = q_inacc; }
  float GetOrigQ() const { return orig_q_; }
  void SetOrigQ(float q) { orig_q_ = q; q_ = q; }
  float GetW() const { return w_; }
  void SetW(float w) { w_ = w; }
  float GetMaxW() const { return max_w_; }
  void SetMaxW(float w) { max_w_ = w; }
  float GetMaxWIncr() const { return max_w_incr_; }
  void SetMaxWIncr(float w) { max_w_incr_ = w; }
  float GetMaxWDecr() const { return max_w_decr_; }
  void SetMaxWDecr(float w) { max_w_decr_ = w; }
  uint16_t GetBranchingInFlight() const { return branching_in_flight_; }
  void SetBranchingInFlight(uint16_t b) { branching_in_flight_ = b; }

  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return is_terminal_; }

  uint16_t GetNumEdges() const { return edges_.size(); }
  uint16_t GetNumChildren() const { return noofchildren_; }
  int16_t GetBestIdx() const { return best_idx_; }
  void SetBestIdx(int16_t idx) { best_idx_ = idx; }
  EdgeGlow* GetEdges() { return edges_.get(); }
  uint16_t GetIndex() const { return index_; }

  uint8_t GetNextUnexpandedEdge() const { return next_unexpanded_edge_; }
  void SetNextUnexpandedEdge(uint8_t n) { next_unexpanded_edge_ = n; }

  uint32_t GetN() const { return n_; }
  void SetN(uint32_t n) { n_ = n; }
  //uint32_t GetNExtendable() const { return n_extendable_; }
  //void SetNExtendable(uint32_t n) { n_extendable_ = n; }

  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result);

  // Deletes all children.
  void ReleaseChildren();

  // Deletes all children except one.
  void ReleaseChildrenExceptOne(NodeGlow* node);

  void IncrParentNumChildren();

//  int ComputeHeight();
//  bool Check();

  // Debug information about the node.
  std::string DebugString() const;

  int CountInternal(uint32_t min_n);
  double QMean(uint32_t min_n);
  double QVariance(uint32_t min_n, double mean);
  double QInaccMean(uint32_t min_n);
  double PMean();
  double PVariance(double mean);
  int LogPCount();
  double LogPMean();
  double LogPVariance(double mean);

private:
	// To minimize the number of padding bytes and to avoid having unnecessary
	// padding when new fields are added, we arrange the fields by size, largest
	// to smallest.

	// TODO: shrink the padding on this somehow? It takes 16 bytes even though
	// only 10 are real! Maybe even merge it into this class??
	EdgeListGlow edges_;

	// 8 byte fields.
	// Pointer to a parent node. nullptr for the root.
	NodeGlow* parent_ = nullptr;

	// 4 byte fields.
	// Average value (from value head of neural network) of all visited nodes in
	// subtree. For terminal nodes, eval is stored. This is from the perspective
	// of the player who "just" moved to reach this position, rather than from the
	// perspective of the player-to-move for the position.
	float q_ = 0.0f;
  float q_inacc_ = 0.0f;
	float orig_q_ = 0.0f;
	float w_ = 0.0f;
	float max_w_incr_ = 0.0f;
	float max_w_decr_ = 0.0f;
	float max_w_ = 0.0f;

	uint32_t n_ = 1;
	//uint32_t n_extendable_ = 0;

	// 2 byte fields.
	// Index of this node is parent's edge list.
	uint16_t index_;

	uint16_t noofchildren_ = 0;
	int16_t best_idx_ = -1;  // index to child where unexpanded edge with highest global weight is
	uint16_t branching_in_flight_ = 0;

	// 1 byte fields.
	// Whether or not this node end game (with a winning of either sides or draw).
	bool is_terminal_ = false;
	uint8_t next_unexpanded_edge_ = 0;

	friend class NodeTreeGlow;
	friend class EdgeGlow;
};

class NodeTreeGlow : public NodeTreeCommon {
 public:
  ~NodeTreeGlow() { DeallocateTree(); }
  // Adds a move to current_head_.
  void MakeMove(Move move);
  // Resets the current head to ensure it doesn't carry over details from a
  // previous search.
  void TrimTreeAtHead();
  // Sets the position in a tree, trying to reuse the tree.
  // If @auto_garbage_collect, old tree is garbage collected immediately. (may
  // take some milliseconds)
  // Returns whether a new position the same game as old position (with some
  // moves added). Returns false, if the position is completely different,
  // or if it's shorter than before.
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<Move>& moves);
  NodeGlow* GetCurrentHead() const { return current_head_; }
  NodeGlow* GetGameBeginNode() const { return gamebegin_node_.get(); }

 private:
  void DeallocateTree();
  // A node which to start search from.
  NodeGlow* current_head_ = nullptr;
  // Root node of a game tree.
  std::unique_ptr<NodeGlow> gamebegin_node_;
};



}  // namespace lczero
