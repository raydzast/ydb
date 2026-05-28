#include "kqp_product_quantization.h"

#include <ydb/core/base/ivf_pq.h>
#include <ydb/core/base/table_index.h>

#include <yql/essentials/minikql/computation/mkql_computation_node_holders.h>
#include <yql/essentials/minikql/computation/mkql_computation_node_impl.h>
#include <yql/essentials/minikql/mkql_node_cast.h>
#include <yql/essentials/minikql/mkql_string_util.h>

namespace NKikimr {
namespace NMiniKQL {

namespace {

class TProductQuantizationBuildDistanceTableWrapper : public TMutableComputationNode<TProductQuantizationBuildDistanceTableWrapper> {
    using TBaseComputation = TMutableComputationNode<TProductQuantizationBuildDistanceTableWrapper>;

public:
    TProductQuantizationBuildDistanceTableWrapper(TComputationMutables& mutables,
        IComputationNode* centroidArg,
        IComputationNode* targetArg,
        IComputationNode* codebookArg,
        IComputationNode* mArg,
        IComputationNode* nbitsArg,
        IComputationNode* settingsArg,
        ui32 segmentMemberIdx,
        ui32 codeMemberIdx,
        ui32 centroidMemberIdx)
        : TBaseComputation(mutables)
        , CentroidArg(centroidArg)
        , TargetArg(targetArg)
        , CodebookArg(codebookArg)
        , MArg(mArg)
        , NbitsArg(nbitsArg)
        , SettingsArg(settingsArg)
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
        const auto settingsValue = SettingsArg->GetValue(ctx);

        const auto settingsBuf = settingsValue.AsStringRef();
        Ydb::Table::VectorIndexSettings vectorSettings;
        MKQL_ENSURE(vectorSettings.ParseFromArray(settingsBuf.Data(), settingsBuf.Size()),
            "ProductQuantizationBuildDistanceTable: failed to parse VectorIndexSettings");

        TVector<TVector<TString>> codebookStorage(m);

        const auto codebookList = CodebookArg->GetValue(ctx);
        const auto iter = codebookList.GetListIterator();
        NUdf::TUnboxedValue rowValue;
        while (iter.Next(rowValue)) {
            const ui32 segment = rowValue.GetElement(SegmentMemberIdx).Get<ui16>();
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
        const TString distanceTable = NIvfPq::BuildDistanceTable(
            residual, codebookView, m, nbits, vectorSettings);
        return MakeString(distanceTable);
    }

private:
    void RegisterDependencies() const final {
        DependsOn(CentroidArg);
        DependsOn(TargetArg);
        DependsOn(CodebookArg);
        DependsOn(MArg);
        DependsOn(NbitsArg);
        DependsOn(SettingsArg);
    }

    IComputationNode* const CentroidArg;
    IComputationNode* const TargetArg;
    IComputationNode* const CodebookArg;
    IComputationNode* const MArg;
    IComputationNode* const NbitsArg;
    IComputationNode* const SettingsArg;
    const ui32 SegmentMemberIdx;
    const ui32 CodeMemberIdx;
    const ui32 CentroidMemberIdx;
};

} // namespace

IComputationNode* WrapProductQuantizationBuildDistanceTable(TCallable& callable, const TComputationNodeFactoryContext& ctx) {
    MKQL_ENSURE(callable.GetInputsCount() == 6, "ProductQuantizationBuildDistanceTable requires exactly 6 arguments");

    const auto* codebookStaticType = callable.GetInput(2).GetStaticType();
    MKQL_ENSURE(codebookStaticType->IsList(), "ProductQuantizationBuildDistanceTable codebook must be a list");
    const auto* listType = static_cast<const TListType*>(codebookStaticType);
    MKQL_ENSURE(listType->GetItemType()->IsStruct(), "ProductQuantizationBuildDistanceTable codebook items must be structs");
    const auto* structType = static_cast<const TStructType*>(listType->GetItemType());

    auto* centroidArg = LocateNode(ctx.NodeLocator, callable, 0);
    auto* targetArg = LocateNode(ctx.NodeLocator, callable, 1);
    auto* codebookArg = LocateNode(ctx.NodeLocator, callable, 2);
    auto* mArg = LocateNode(ctx.NodeLocator, callable, 3);
    auto* nbitsArg = LocateNode(ctx.NodeLocator, callable, 4);
    auto* settingsArg = LocateNode(ctx.NodeLocator, callable, 5);

    return new TProductQuantizationBuildDistanceTableWrapper(ctx.Mutables,
        centroidArg, targetArg, codebookArg, mArg, nbitsArg, settingsArg,
        structType->GetMemberIndex(NTableIndex::NIvfPq::SubspaceColumn),
        structType->GetMemberIndex(NTableIndex::NIvfPq::CellColumn),
        structType->GetMemberIndex(NTableIndex::NIvfPq::CentroidColumn));
}

namespace {

class TProductQuantizationEncodeWrapper : public TMutableComputationNode<TProductQuantizationEncodeWrapper> {
    using TBaseComputation = TMutableComputationNode<TProductQuantizationEncodeWrapper>;

public:
    TProductQuantizationEncodeWrapper(TComputationMutables& mutables,
        IComputationNode* residualArg,
        IComputationNode* codebookArg,
        IComputationNode* mArg,
        IComputationNode* nbitsArg,
        IComputationNode* settingsArg,
        ui32 subspaceMemberIdx,
        ui32 cellMemberIdx,
        ui32 centroidMemberIdx)
        : TBaseComputation(mutables)
        , ResidualArg(residualArg)
        , CodebookArg(codebookArg)
        , MArg(mArg)
        , NbitsArg(nbitsArg)
        , SettingsArg(settingsArg)
        , SubspaceMemberIdx(subspaceMemberIdx)
        , CellMemberIdx(cellMemberIdx)
        , CentroidMemberIdx(centroidMemberIdx)
    {
    }

    NUdf::TUnboxedValue DoCalculate(TComputationContext& ctx) const {
        const auto residualValue = ResidualArg->GetValue(ctx);
        const ui32 m = MArg->GetValue(ctx).Get<ui32>();
        const ui32 nbits = NbitsArg->GetValue(ctx).Get<ui32>();
        const auto settingsValue = SettingsArg->GetValue(ctx);

        const auto settingsBuf = settingsValue.AsStringRef();
        Ydb::Table::VectorIndexSettings settings;
        MKQL_ENSURE(settings.ParseFromArray(settingsBuf.Data(), settingsBuf.Size()),
            "ProductQuantizationEncode: failed to parse VectorIndexSettings");

        TVector<TVector<TString>> codebookCentroids(m);

        const auto codebookList = CodebookArg->GetValue(ctx);
        const auto iter = codebookList.GetListIterator();
        NUdf::TUnboxedValue rowValue;
        while (iter.Next(rowValue)) {
            const ui32 subspace = rowValue.GetElement(SubspaceMemberIdx).Get<ui16>();
            const ui32 cell = rowValue.GetElement(CellMemberIdx).Get<ui16>();
            const auto subCentroidValue = rowValue.GetElement(CentroidMemberIdx);

            MKQL_ENSURE(subspace < m, "ProductQuantizationEncode: codebook subspace index >= m");

            auto& subspaceVec = codebookCentroids[subspace];
            if (subspaceVec.size() <= cell) {
                subspaceVec.resize(cell + 1);
            }
            subspaceVec[cell] = TString(subCentroidValue.AsStringRef());
        }

        TString error;
        auto pq = NIvfPq::TProductQuantizer::Create(m, settings, /* maxRounds */ 0, error);
        MKQL_ENSURE(pq != nullptr, "ProductQuantizationEncode: failed to create TProductQuantizer: " << error);

        for (ui32 i = 0; i < m; ++i) {
            MKQL_ENSURE(!codebookCentroids[i].empty(),
                "ProductQuantizationEncode: codebook subspace " << i << " is empty");
            const bool ok = pq->SetSubquantizerCentroids(i, std::move(codebookCentroids[i]));
            MKQL_ENSURE(ok, "ProductQuantizationEncode: failed to set subquantizer centroids for subspace " << i);
        }

        const auto cells = pq->Quantize(residualValue.AsStringRef());
        MKQL_ENSURE(cells.size() == m, "ProductQuantizationEncode: Quantize produced " << cells.size()
            << " cells, expected " << m);

        TVector<ui16> codeCells(cells.begin(), cells.end());
        const TString code = NIvfPq::NPackedNBitVector::Serialize(codeCells, nbits);
        return MakeString(code);
    }

private:
    void RegisterDependencies() const final {
        DependsOn(ResidualArg);
        DependsOn(CodebookArg);
        DependsOn(MArg);
        DependsOn(NbitsArg);
        DependsOn(SettingsArg);
    }

    IComputationNode* const ResidualArg;
    IComputationNode* const CodebookArg;
    IComputationNode* const MArg;
    IComputationNode* const NbitsArg;
    IComputationNode* const SettingsArg;
    const ui32 SubspaceMemberIdx;
    const ui32 CellMemberIdx;
    const ui32 CentroidMemberIdx;
};

} // namespace

IComputationNode* WrapProductQuantizationEncode(TCallable& callable, const TComputationNodeFactoryContext& ctx) {
    MKQL_ENSURE(callable.GetInputsCount() == 5, "ProductQuantizationEncode requires exactly 5 arguments");

    const auto* codebookStaticType = callable.GetInput(1).GetStaticType();
    MKQL_ENSURE(codebookStaticType->IsList(), "ProductQuantizationEncode codebook must be a list");
    const auto* listType = static_cast<const TListType*>(codebookStaticType);
    MKQL_ENSURE(listType->GetItemType()->IsStruct(), "ProductQuantizationEncode codebook items must be structs");
    const auto* structType = static_cast<const TStructType*>(listType->GetItemType());

    auto* residualArg = LocateNode(ctx.NodeLocator, callable, 0);
    auto* codebookArg = LocateNode(ctx.NodeLocator, callable, 1);
    auto* mArg = LocateNode(ctx.NodeLocator, callable, 2);
    auto* nbitsArg = LocateNode(ctx.NodeLocator, callable, 3);
    auto* settingsArg = LocateNode(ctx.NodeLocator, callable, 4);

    return new TProductQuantizationEncodeWrapper(ctx.Mutables,
        residualArg, codebookArg, mArg, nbitsArg, settingsArg,
        structType->GetMemberIndex(NTableIndex::NIvfPq::SubspaceColumn),
        structType->GetMemberIndex(NTableIndex::NIvfPq::CellColumn),
        structType->GetMemberIndex(NTableIndex::NIvfPq::CentroidColumn));
}

} // namespace NMiniKQL
} // namespace NKikimr
