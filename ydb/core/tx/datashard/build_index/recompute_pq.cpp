#include "common_helper.h"
#include "kmeans_helper.h"

#include <ydb/core/base/ivf_pq.h>
#include <ydb/core/tx/datashard/datashard_impl.h>
#include <ydb/core/tx/datashard/datashard.h>

#include <ydb/core/base/kmeans_clusters.h>
#include <ydb/core/base/table_index.h>
#include <ydb/library/yql/udfs/common/knn/knn-serializer-shared.h>

#include <memory>

namespace NKikimr::NDataShard {
using NKikimr::NIvfPq::TProductQuantizer;
using namespace NKMeans;

class TRecomputePqScan : public TActor<TRecomputePqScan>, public IActorExceptionHandler, public NTable::IScan {
protected:
    const ui64 TabletId = 0;
    const ui64 BuildId = 0;
    const TAutoPtr<TEvDataShard::TEvRecomputePqResponse> Response;
    const TActorId ResponseActorId;

    TTags ScanTags;
    NTable::TPos EmbeddingPos = 0;

    ui64 ReadRows = 0;
    ui64 ReadBytes = 0;

    bool InForeign = false;
    NTable::TPos IsForeignPos = 0;

    IDriver* Driver = nullptr;
    NYql::TIssues Issues;

    TLead Lead;

    std::unique_ptr<TProductQuantizer> ProductQuantizer;
    const ui64 M;

public:
    static constexpr NKikimrServices::TActivity::EType ActorActivityType() {
        return NKikimrServices::TActivity::RECOMPUTE_PQ_SCAN_ACTOR;
    }

    TRecomputePqScan(ui64 tabletId, const TUserTable& table, TLead&& lead,
        const NKikimrTxDataShard::TEvRecomputePqRequest& request,
        const TActorId& responseActorId, TAutoPtr<TEvDataShard::TEvRecomputePqResponse>&& response,
        std::unique_ptr<TProductQuantizer>&& productQuantizer)
        : TActor(&TThis::StateWork)
        , TabletId(tabletId)
        , BuildId(request.GetId())
        , Response(std::move(response))
        , ResponseActorId(responseActorId)
        , Lead(std::move(lead))
        , ProductQuantizer(std::move(productQuantizer))
        , M(ProductQuantizer->SubspaceCount)
    {
        LOG_I("Create " << Debug());

        // InForeign = request.GetSkipOverlapForeign();

        const auto& embedding = request.GetEmbeddingColumn();
        NTable::TPos dataPos = 0;
        ScanTags = MakeScanTags(table, embedding, {}, false, EmbeddingPos, dataPos, InForeign ? &IsForeignPos : nullptr);
        Lead.SetTags(ScanTags);
    }

    TInitialState Prepare(IDriver* driver, TIntrusiveConstPtr<TScheme>) final {
        TActivationContext::AsActorContext().RegisterWithSameMailbox(this);
        LOG_I("Prepare " << Debug());

        Driver = driver;

        return {EScan::Feed, {}};
    }

    TAutoPtr<IDestructable> Finish(const std::exception& exc) final {
        Issues.AddIssue(NYql::TIssue(TStringBuilder()
            << "Scan failed " << exc.what()));
        return Finish(EStatus::Exception);
    }

    TAutoPtr<IDestructable> Finish(EStatus status) final {
        auto& record = Response->Record;
        record.MutableMeteringStats()->SetReadRows(ReadRows);
        record.MutableMeteringStats()->SetReadBytes(ReadBytes);
        record.MutableMeteringStats()->SetCpuTimeUs(Driver->GetTotalCpuTimeUs());

        if (status == EStatus::Exception) {
            record.SetStatus(NKikimrIndexBuilder::EBuildStatus::BUILD_ERROR);
        } else if (status != NTable::EStatus::Done) {
            record.SetStatus(NKikimrIndexBuilder::EBuildStatus::ABORTED);
        } else {
            record.SetStatus(NKikimrIndexBuilder::EBuildStatus::DONE);
            FillResponse();
        }

        NYql::IssuesToMessage(Issues, record.MutableIssues());

        if (Response->Record.GetStatus() == NKikimrIndexBuilder::DONE) {
            LOG_N("Done " << Debug() << " " << ToShortDebugString(Response->Record));
        } else {
            LOG_E("Failed " << Debug() << " " << ToShortDebugString(Response->Record));
        }
        Send(ResponseActorId, Response.Release());

        Driver = nullptr;
        this->PassAway();
        return nullptr;
    }

    bool OnUnhandledException(const std::exception& exc) final {
        if (!Driver) {
            return false;
        }
        Driver->Throw(exc);
        return true;
    }

    void Describe(IOutputStream& out) const final {
        out << Debug();
    }

    EScan Seek(TLead& lead, ui64 seq) final {
        LOG_T("Seek " << seq << " " << Debug());

        lead = Lead;

        return EScan::Feed;
    }

    EScan Feed(TArrayRef<const TCell> key, const TRow& row) final {
        // LOG_T("Feed " << Debug());

        ++ReadRows;
        ReadBytes += CountRowCellBytes(key, *row);

        Feed(key, *row);

        return EScan::Feed;
    }

    EScan Exhausted() final {
        LOG_T("Exhausted " << Debug());

        return EScan::Final;
    }

protected:
    STFUNC(StateWork) {
        switch (ev->GetTypeRewrite()) {
            default:
                LOG_E("StateWork unexpected event type: " << ev->GetTypeRewrite()
                    << " event: " << ev->ToString() << " " << Debug());
        }
    }

    TString Debug() const {
        return TStringBuilder{}
            << "TRecomputePqScan TabletId: " << TabletId
            << " Id: " << BuildId
            << " ProductQuantizer: " << ProductQuantizer->Debug();
    }

    void Feed(TArrayRef<const TCell>, TArrayRef<const TCell> row) {
        if (InForeign) {
            bool foreign = row.at(IsForeignPos).AsValue<bool>();
            if (foreign) {
                // Skip rows from "non-domestic" clusters to not affect K-means centroids
                return;
            }
        }

        ProductQuantizer->Aggregate(row[EmbeddingPos].AsBuf());
    }

    void FillResponse() {
        auto& record = Response->Record;
        ProductQuantizer->Recompute();

        for (size_t i = 0; i < M; ++i) {
            const auto& outClusters = ProductQuantizer->GetSubspaceCentroids(i);
            const auto& outSizes = ProductQuantizer->GetSubspaceNextClusterSizes(i);

            auto* v = record.AddSubquantizerResults();
            *v->MutableCentroids() = {outClusters.begin(), outClusters.end()};
            *v->MutableClusterSizes() = {outSizes.begin(), outSizes.end()};
        }
        record.SetStatus(NKikimrIndexBuilder::EBuildStatus::DONE);
    }
};

class TDataShard::TTxHandleSafeRecomputePqScan final: public NTabletFlatExecutor::TTransactionBase<TDataShard> {
public:
    TTxHandleSafeRecomputePqScan(TDataShard* self, TEvDataShard::TEvRecomputePqRequest::TPtr&& ev)
        : TTransactionBase(self)
        , Ev(std::move(ev))
    {
    }

    bool Execute(TTransactionContext&, const TActorContext& ctx) final {
        Self->HandleSafe(Ev, ctx);
        return true;
    }

    void Complete(const TActorContext&) final {
    }

private:
    TEvDataShard::TEvRecomputePqRequest::TPtr Ev;
};

void TDataShard::Handle(TEvDataShard::TEvRecomputePqRequest::TPtr& ev, const TActorContext&) {
    Execute(new TTxHandleSafeRecomputePqScan(this, std::move(ev)));
}

void TDataShard::HandleSafe(TEvDataShard::TEvRecomputePqRequest::TPtr& ev, const TActorContext& ctx) {
    auto& request = ev->Get()->Record;
    const ui64 id = request.GetId();
    auto rowVersion = request.HasSnapshotStep() || request.HasSnapshotTxId()
        ? TRowVersion(request.GetSnapshotStep(), request.GetSnapshotTxId())
        : GetMvccTxVersion(EMvccTxMode::ReadOnly);
    TScanRecord::TSeqNo seqNo = {request.GetSeqNoGeneration(), request.GetSeqNoRound()};

    try {
        auto response = MakeHolder<TEvDataShard::TEvRecomputePqResponse>();
        FillScanResponseCommonFields(*response, id, TabletID(), seqNo);

        LOG_N("Starting TRecomputePqScan TabletId: " << TabletID()
            << " " << ToShortDebugString(request)
            << " row version " << rowVersion);

        // Note: it's very unlikely that we have volatile txs before this snapshot
        if (VolatileTxManager.HasVolatileTxsAtSnapshot(rowVersion)) {
            VolatileTxManager.AttachWaitingSnapshotEvent(rowVersion, std::unique_ptr<IEventHandle>(ev.Release()));
            return;
        }

        auto badRequest = [&](const TString& error) {
            response->Record.SetStatus(NKikimrIndexBuilder::EBuildStatus::BAD_REQUEST);
            auto issue = response->Record.AddIssues();
            issue->set_severity(NYql::TSeverityIds::S_ERROR);
            issue->set_message(error);
        };
        auto trySendBadRequest = [&] {
            if (response->Record.GetStatus() == NKikimrIndexBuilder::EBuildStatus::BAD_REQUEST) {
                LOG_E("Rejecting TRecomputePqScan bad request TabletId: " << TabletID()
                    << " " << ToShortDebugString(request)
                    << " with response " << ToShortDebugString(response->Record));
                ctx.Send(ev->Sender, std::move(response));
                return true;
            } else {
                return false;
            }
        };

        // 1. Validating table and path existence
        if (request.GetTabletId() != TabletID()) {
            badRequest(TStringBuilder() << "Wrong shard " << request.GetTabletId() << " this is " << TabletID());
        }
        if (!IsStateActive()) {
            badRequest(TStringBuilder() << "Shard " << TabletID() << " is " << State << " and not ready for requests");
        }
        const auto pathId = TPathId::FromProto(request.GetPathId());
        const auto* userTableIt = GetUserTables().FindPtr(pathId.LocalPathId);
        if (!userTableIt) {
            badRequest(TStringBuilder() << "Unknown table id: " << pathId.LocalPathId);
        }
        if (trySendBadRequest()) {
            return;
        }
        const auto& userTable = **userTableIt;

        // 2. Validating request fields
        if (request.HasSnapshotStep() || request.HasSnapshotTxId()) {
            const TSnapshotKey snapshotKey(pathId, rowVersion.Step, rowVersion.TxId);
            if (!SnapshotManager.FindAvailable(snapshotKey)) {
                badRequest(TStringBuilder() << "Unknown snapshot for path id " << pathId.OwnerId << ":" << pathId.LocalPathId
                    << ", snapshot step is " << snapshotKey.Step << ", snapshot tx is " << snapshotKey.TxId);
            }
        }

        NTable::TLead lead;
        if (request.HasParent() && request.GetParent() == 0) {
            badRequest("Invalid parent: 0");
        } else if (request.HasParent()) {
            Y_ASSERT(request.GetParent() != 0);

            TCell from, to;
            const auto range = CreateRangeFrom(userTable, request.GetParent(), from, to);
            if (range.IsEmptyRange(userTable.KeyColumnTypes)) {
                badRequest(TStringBuilder() << " requested range doesn't intersect with table range");
            }
            lead = CreateLeadFrom(range);;
        } else {
            lead.To({}, NTable::ESeek::Lower);
        }

        auto tags = GetAllTags(userTable);
        if (!tags.contains(request.GetEmbeddingColumn())) {
            badRequest(TStringBuilder() << "Unknown embedding column: " << request.GetEmbeddingColumn());
        }

        // 3. Validating vector index settings
        TString error;
        auto productQuantizer = TProductQuantizer::Create(request.GetM(), request.GetSettings(), 0, error);
        if (!productQuantizer) {
            badRequest(error);
        } else if (request.SubquantizersSize() != request.GetM()) {
            badRequest(TStringBuilder() << "Invalid subquantizers count: " << request.SubquantizersSize() << " expected " << request.GetM());
        } else {
            for (size_t i = 0; i < request.SubquantizersSize(); ++i) {
                const auto& centroids = request.GetSubquantizers(i).GetCentroids();
                if (!productQuantizer->SetSubquantizerCentroids(i, {centroids.begin(), centroids.end()})) {
                    badRequest(TStringBuilder() << "Failed to set clusters for subquantizer " << i << ": Clusters have invalid format");
                    break;
                }
            }
        }

        if (trySendBadRequest()) {
            return;
        }

        TAutoPtr<NTable::IScan> scan = new TRecomputePqScan(
            TabletID(), userTable, std::move(lead), request, ev->Sender,
            std::move(response), std::move(productQuantizer)
        );

        StartScan(this, std::move(scan), id, seqNo, rowVersion, userTable.LocalTid);
    } catch (const std::exception& exc) {
        FailScan<TEvDataShard::TEvRecomputePqResponse>(id, TabletID(), ev->Sender, seqNo, exc, "TRecomputePqScan");
    }
}

}
