#pragma once

#include <yql/essentials/minikql/computation/mkql_computation_node.h>

namespace NKikimr {
namespace NMiniKQL {

IComputationNode* WrapKqpBuildPqDistanceTable(TCallable& callable, const TComputationNodeFactoryContext& ctx);

IComputationNode* WrapKqpPqEncode(TCallable& callable, const TComputationNodeFactoryContext& ctx);

} // namespace NMiniKQL
} // namespace NKikimr
