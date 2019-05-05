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

#include "glow/strategy.h"

namespace lczero {



void set_strategy_parameters(const SearchParams *params) {
}

float compute_q_and_weights(NodeGlow *node, int node_n) {
	float sum_p = 0.0;
	float max_q = -1.0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
		sum_p += node->GetEdges()[i->GetIndex()].GetP();
		float q = i->GetQ();
		if (q > max_q) max_q = q;
	}

	float sum_w = 0.0;
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
		float w = exp(35.0 * (i->GetQ() - max_q));
		i->SetW(w);
		sum_w += w;
	}

	float sum_ratio = sum_p / sum_w;
	float q = (1.0 - sum_p) * node->GetOrigQ();
  for (NodeGlow *i = node->GetFirstChild(); i != nullptr; i = i->GetNextSibling()) {
		float w = i->GetW() * sum_ratio;
		i->SetW(w);
		q -= w * i->GetQ();
	}

	return q;
}

}  // namespace lczero
