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
 */

#pragma once

#include <vector>
#include "proto/net.pb.h"

namespace lczero {

// DEPRECATED! DEPRECATED! DEPRECATED! DEPRECATED! DEPRECATED! DEPRECATED!!!
// Legacy structure describing network weights.
// Please try to migrate away from this struture do not add anything new
// to it.

struct LegacyWeights {
  explicit LegacyWeights(const pblczero::Weights& weights);

  using Vec = std::vector<float>;
  struct ConvBlock {
    explicit ConvBlock(const pblczero::Weights::ConvBlock& block);
    // Invert the bn_stddivs elements of a ConvBlock.
    void InvertStddev();
    // Offset bn_means by biases of a ConvBlock.
    void OffsetMeans();
    // Return a vector of inverted bn_stddivs of a ConvBlock.
    std::vector<float> GetInvertedStddev() const;
    // Return a vector of bn_means offset by biases of a ConvBlock.
    std::vector<float> GetOffsetMeans() const;

    Vec weights;
    Vec biases;
    Vec bn_gammas;
    Vec bn_betas;
    Vec bn_means;
    Vec bn_stddivs;
  };

  struct SEunit {
    explicit SEunit(const pblczero::Weights::SEunit& se);
    Vec w1;
    Vec b1;
    Vec w2;
    Vec b2;
  };

  struct Residual {
    explicit Residual(const pblczero::Weights::Residual& residual);
    ConvBlock conv1;
    ConvBlock conv2;
    SEunit se;
    bool has_se;
  };

  // Input convnet.
  ConvBlock input;

  // Residual tower.
  std::vector<Residual> residual;

  // Policy head
  ConvBlock policy;
  Vec ip_pol_w;
  Vec ip_pol_b;

  // Value head
  ConvBlock value;
  Vec ip1_val_w;
  Vec ip1_val_b;
  Vec ip2_val_w;
  Vec ip2_val_b;
};

}  // namespace lczero
