#include "kqp_pq_distance_table.h"

#include <ydb/core/base/ivf_pq.h>
#include <ydb/core/base/table_index.h>

#include <yql/essentials/minikql/computation/mkql_computation_node_holders.h>
#include <yql/essentials/minikql/computation/mkql_computation_node_impl.h>
#include <yql/essentials/minikql/mkql_node_cast.h>
#include <yql/essentials/minikql/mkql_string_util.h>

namespace NKikimr {
namespace NMiniKQL {

namespace {

class TKqpBuildPqDistanceTableWrapper : public TMutableComputationNode<TKqpBuildPqDistanceTableWrapper> {
    using TBaseComputation = TMutableComputationNode<TKqpBuildPqDistanceTableWrapper>;

public:
    TKqpBuildPqDistanceTableWrapper(TComputationMutables& mutables,
        IComputationNode* centroidArg,
        IComputationNode* targetArg,
        IComputationNode* codebookArg,
        IComputationNode* mArg,
        IComputationNode* nbitsArg,
        ui32 segmentMemberIdx,
        ui32 codeMemberIdx,
        ui32 centroidMemberIdx)
        : TBaseComputation(mutables)
        , CentroidArg(centroidArg)
        , TargetArg(targetArg)
        , CodebookArg(codebookArg)
        , MArg(mArg)
        , NbitsArg(nbitsArg)
        , SegmentMemberIdx(segmentMemberIdx)
        , CodeMemberIdx(codeMemberIdx)
        , CentroidMemberIdx(centroidMemberIdx)
    {
    }

    NUdf::TUnboxedValue DoCalculate(TComputationContext& ctx) const {
        const auto centroidValue = CentroidArg->GetValue(ctx);
        const auto targetValue = TargetArg->GetValue(ctx);
        const ui32 m = MArg->GetValue(ctx).Get<ui32>();
        const ui32 nbits = NbitsArg->GetValue(ctx).Get<ui32>();

        // Materialize sparse codebook into TVector<TVector<TString>> first, so the
        // backing storage outlives the TStringBuf views passed to BuildPqDistanceTable.
        // codebook[mIdx].size() == max(Code per segment) + 1; missing trailing slots
        // are padded with +Inf inside NIvfPq::BuildPqDistanceTable (Stage 1d).
        TVector<TVector<TString>> codebookStorage(m);

        const auto codebookList = CodebookArg->GetValue(ctx);
        const auto iter = codebookList.GetListIterator();
        NUdf::TUnboxedValue rowValue;
        while (iter.Next(rowValue)) {
            const ui32 segment = rowValue.GetElement(SegmentMemberIdx).Get<ui8>();
            const ui32 code = rowValue.GetElement(CodeMemberIdx).Get<ui16>();
            const auto subCentroidValue = rowValue.GetElement(CentroidMemberIdx);

            MKQL_ENSURE(segment < m, "Codebook segment index >= m");

            auto& segmentVec = codebookStorage[segment];
            if (segmentVec.size() <= code) {
                segmentVec.resize(code + 1);
            }
            segmentVec[code] = TString(subCentroidValue.AsStringRef());
        }

        TVector<TVector<TStringBuf>> codebookView(m);
        for (ui32 i = 0; i < m; ++i) {
            codebookView[i].reserve(codebookStorage[i].size());
            for (const auto& s : codebookStorage[i]) {
                codebookView[i].emplace_back(s);
            }
        }

        const TString residual = NIvfPq::SubtractCentroid(targetValue.AsStringRef(), centroidValue.AsStringRef());
        const TString distanceTable = NIvfPq::BuildPqDistanceTable(residual, codebookView, m, nbits);
        return MakeString(distanceTable);
    }

private:
    void RegisterDependencies() const final {
        DependsOn(CentroidArg);
        DependsOn(TargetArg);
        DependsOn(CodebookArg);
        DependsOn(MArg);
        DependsOn(NbitsArg);
    }

    IComputationNode* const CentroidArg;
    IComputationNode* const TargetArg;
    IComputationNode* const CodebookArg;
    IComputationNode* const MArg;
    IComputationNode* const NbitsArg;
    const ui32 SegmentMemberIdx;
    const ui32 CodeMemberIdx;
    const ui32 CentroidMemberIdx;
};

} // namespace

IComputationNode* WrapKqpBuildPqDistanceTable(TCallable& callable, const TComputationNodeFactoryContext& ctx) {
    MKQL_ENSURE(callable.GetInputsCount() == 5, "KqpBuildPqDistanceTable requires exactly 5 arguments");

    const auto* codebookStaticType = callable.GetInput(2).GetStaticType();
    MKQL_ENSURE(codebookStaticType->IsList(), "KqpBuildPqDistanceTable codebook must be a list");
    const auto* listType = static_cast<const TListType*>(codebookStaticType);
    MKQL_ENSURE(listType->GetItemType()->IsStruct(), "KqpBuildPqDistanceTable codebook items must be structs");
    const auto* structType = static_cast<const TStructType*>(listType->GetItemType());

    auto* centroidArg = LocateNode(ctx.NodeLocator, callable, 0);
    auto* targetArg = LocateNode(ctx.NodeLocator, callable, 1);
    auto* codebookArg = LocateNode(ctx.NodeLocator, callable, 2);
    auto* mArg = LocateNode(ctx.NodeLocator, callable, 3);
    auto* nbitsArg = LocateNode(ctx.NodeLocator, callable, 4);

    return new TKqpBuildPqDistanceTableWrapper(ctx.Mutables,
        centroidArg, targetArg, codebookArg, mArg, nbitsArg,
        structType->GetMemberIndex(NTableIndex::NIvfPq::SubspaceColumn),
        structType->GetMemberIndex(NTableIndex::NIvfPq::CellColumn),
        structType->GetMemberIndex(NTableIndex::NIvfPq::CentroidColumn));
}

} // namespace NMiniKQL
} // namespace NKikimr
