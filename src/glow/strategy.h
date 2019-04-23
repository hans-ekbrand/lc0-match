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

#include "search/params.h"
#include "glow/node.h"

namespace lczero {

void set_strategy_parameters(const SearchParams *params);

// compute q and w:s for a node
// inputs:
//  node - the node
//  q of node's children
//  orig_q of node
//  p of node's edges
//  n of node
//  n of node's children
// outputs:
//  return new q (note: node's q should not be changed, that's up to function caller)
//  w of node's children
float compute_q_and_weights(NodeGlow *node);


}  // namespace lczero
