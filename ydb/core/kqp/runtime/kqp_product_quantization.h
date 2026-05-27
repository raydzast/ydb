#pragma once

#include <yql/essentials/minikql/computation/mkql_computation_node.h>

namespace NKikimr {
namespace NMiniKQL {

IComputationNode* WrapProductQuantizationBuildDistanceTable(TCallable& callable, const TComputationNodeFactoryContext& ctx);

IComputationNode* WrapProductQuantizationEncode(TCallable& callable, const TComputationNodeFactoryContext& ctx);

} // namespace NMiniKQL
} // namespace NKikimr
