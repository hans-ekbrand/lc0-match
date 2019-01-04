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

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/position.h"
#include "neural/writer.h"
#include "utils/mutex.h"

namespace lczero {


class Node_revamp;
class Edge_revamp {
 public:
  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  Node_revamp* GetChild() { return child_.get(); }
  void CreateChild(Node_revamp* parent, uint16_t index);

  void ReleaseChild();
  void ReleaseChildIfIsNot(Node_revamp* node_to_save);

  // Debug information about the edge.
  std::string DebugString() const;

 private:
  void SetMove(Move move) { move_ = move; }

  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Pointer to child of this edge. nullptr for no node.
  std::unique_ptr<Node_revamp> child_ = nullptr;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;

  friend class EdgeList_revamp;
  friend class Node_revamp;
};

// Array of Edges.
class EdgeList_revamp {
 public:
  EdgeList_revamp() {}
  EdgeList_revamp(MoveList moves);
  Edge_revamp* get() const { return edges_.get(); }
  Edge_revamp& operator[](size_t idx) const { return edges_[idx]; }
  operator bool() const { return static_cast<bool>(edges_); }
  uint16_t size() const { return size_; }

 private:
  std::unique_ptr<Edge_revamp[]> edges_;
  uint16_t size_ = 0;
};

class Node_revamp {
 public:
  // Takes pointer to a parent node and own index in a parent.
  Node_revamp(Node_revamp* parent, uint16_t index) : parent_(parent), index_(index) {}

  // Allocates a new edge and a new node. The node has to be no edges before
  // that.
  Node_revamp* CreateSingleChildNode(Move m);

  // Creates edges from a movelist. There has to be no edges before that.
  void CreateEdges(const MoveList& moves);
  
  void SortEdgesByPValue();

  // Gets parent node.
  Node_revamp* GetParent() const { return parent_; }

  // Returns whether a node has children.
  bool HasChildren() const { return edges_; }

  // Returns node eval, i.e. average subtree V for non-terminal node and -1/0/1
  // for terminal nodes.
  float GetQ() const { return q_; }
  void SetQ(float q) { q_ = q; }

  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return is_terminal_; }
  uint16_t GetNumEdges() const { return edges_.size(); }
  uint16_t GetNumChildren() const { return noofchildren_; }
  Edge_revamp* GetEdges() { return edges_.get(); }
  
  uint32_t GetN() const { return n_; }
  void IncreaseN(uint32_t dn) { n_ += dn; }

  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result);

  // Updates max depth, if new depth is larger.
  void UpdateMaxDepth(int depth);

  // Calculates the full depth if new depth is larger, updates it, returns
  // in depth parameter, and returns true if it was indeed updated.
  bool UpdateFullDepth(uint16_t* depth);

  // Deletes all children.
  void ReleaseChildren();

  // Deletes all children except one.
  void ReleaseChildrenExceptOne(Node_revamp* node);

  // For a child node, returns corresponding edge.
  Edge_revamp* GetEdgeToNode(const Node_revamp* node) const;

  Node_revamp* GetNextLeaf(const Node_revamp* root, PositionHistory *history);
  void ExtendNode(PositionHistory* history);

  int ComputeHeight();

  // Debug information about the node.
  std::string DebugString() const;

 private:
  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.

  // TODO: shrink the padding on this somehow? It takes 16 bytes even though
  // only 10 are real! Maybe even merge it into this class??
  EdgeList_revamp edges_;

  // 8 byte fields.
  // Pointer to a parent node. nullptr for the root.
  Node_revamp* parent_ = nullptr;

  // 4 byte fields.
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored. This is from the perspective
  // of the player who "just" moved to reach this position, rather than from the
  // perspective of the player-to-move for the position.
  float q_ = 0.0f;

  uint32_t n_ = 1;

  // 2 byte fields.
  // Index of this node is parent's edge list.
  uint16_t index_;
  
  uint16_t noofchildren_ = 0;

  // 1 byte fields.
  // Whether or not this node end game (with a winning of either sides or draw).
  bool is_terminal_ = false;

  // TODO(mooskagh) Unfriend NodeTree_revamp.
  friend class NodeTree_revamp;
  friend class Edge_revamp;
};

class NodeTree_revamp {
 public:
  ~NodeTree_revamp() { DeallocateTree(); }
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
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node_revamp* GetCurrentHead() const { return current_head_; }
  Node_revamp* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  void DeallocateTree();
  // A node which to start search from.
  Node_revamp* current_head_ = nullptr;
  // Root node of a game tree.
  std::unique_ptr<Node_revamp> gamebegin_node_;
  PositionHistory history_;
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
class Edge_revamp {
 public:
  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

};

// Array of Edges.
class EdgeList_revamp {
 public:
  EdgeList_revamp() {}
  EdgeList_revamp(MoveList moves);
  Edge_revamp* get() const { return edges_.get(); }
  Edge_revamp& operator[](size_t idx) const { return edges_[idx]; }
  operator bool() const { return static_cast<bool>(edges_); }
  uint16_t size() const { return size_; }

 private:
  std::unique_ptr<Edge_revamp[]> edges_;
  uint16_t size_ = 0;
};


class Node_revamp {
 public:
  // 8 byte fields.
  // Pointer to a parent node. nullptr for the root.
  Node_revamp *parent_ = nullptr;
  // Pointer to a first child. nullptr for a leaf node.
  std::unique_ptr<Node_revamp> child_;
  // Pointer to a next sibling. nullptr if there are no further siblings.
  std::unique_ptr<Node_revamp> sibling_;

  // TODO: shrink the padding on this somehow? It takes 16 bytes even though
  // only 10 are real! Maybe even merge it into this class??
  EdgeList_revamp edges_;
  
};

class NodeTree_revamp {
 public:
//  ~NodeTree_revamp() { DeallocateTree(); }
  // Adds a move to current_head_.
//  void MakeMove(Move move);
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<Move>& moves);
//  const Position& HeadPosition() const { return history_.Last(); }
//  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  int GetPlyCount() const { return 1; }
//  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  bool IsBlackToMove() const { return false; }

  // Root node of a game tree.
  std::unique_ptr<Node_revamp> gamebegin_node_;
  PositionHistory history_;

 private:
//  void DeallocateTree();
  // A node which to start search from.
//  Node_revamp* current_head_ = nullptr;
  // Root node of a game tree.
//  std::unique_ptr<Node_revamp> gamebegin_node_;
//  PositionHistory history_;
};
*/

}  // namespace lczero
