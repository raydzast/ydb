#include "common_helper.h"
#include "kmeans_helper.h"

#include <ydb/core/tx/datashard/datashard_impl.h>
#include <ydb/core/scheme/scheme_types_proto.h>

#include <ydb/core/base/ivf_pq.h>

namespace NKikimr::NDataShard {
using NKikimr::NIvfPq::TProductQuantizer;

namespace {

    void FillLeadFromRange(const TTableRange& range, NTable::TLead& lead) {
        lead.To(range.From, range.InclusiveFrom ? NTable::ESeek::Lower : NTable::ESeek::Upper);
        lead.Until(range.To, range.InclusiveTo);
    }

    // TODO(raydzast): move to some place like kmeans_helper.h
    // TODO(raydzast): add support for foreign columns
    std::shared_ptr<NTxProxy::TUploadTypes> MakePqOutputTypes(
        const TUserTable& table, const google::protobuf::RepeatedPtrField<TProtoStringType>& data,
        bool withParent
        // bool withForeignFlag
    ) {
        auto types = GetAllTypes(table);

        auto result = std::make_shared<NTxProxy::TUploadTypes>();

        // Key columns
        Ydb::Type type;
        if (withParent) {
            type.set_type_id(NTableIndex::NIvfPq::ClusterIdType);
            result->emplace_back(NTableIndex::NIvfPq::ParentColumn, type);
        }
        // if (!withForeignFlag) {
        // }

        auto addType = [&](const auto& column) {
            auto it = types.find(column);
            if (it != types.end()) {
                NScheme::ProtoFromTypeInfo(it->second, type);
                result->emplace_back(it->first, type);
                types.erase(it);
            }
        };
        
        for (const auto& column : table.KeyColumnIds) {
            addType(table.Columns.at(column).Name);
        }

        // if (withForeignFlag) {
        //     type.set_type_id(NTableIndex::NKMeans::ClusterIdType);
        //     result->emplace_back(NTableIndex::NKMeans::ParentColumn, type);
        //     type.set_type_id(NTableIndex::NKMeans::IsForeignType);
        //     result->emplace_back(NTableIndex::NKMeans::IsForeignColumn, type);
        //     type.set_type_id(NTableIndex::NKMeans::DistanceType);
        //     result->emplace_back(NTableIndex::NKMeans::DistanceColumn, type);
        // }

        type.set_type_id(NTableIndex::NIvfPq::CodeType);
        result->emplace_back(NTableIndex::NIvfPq::CodeColumn, type);

        for (const auto& column : data) {
            addType(column);
        }

        return result;
    }

}

class TEncodePqScan : public TActor<TEncodePqScan>, public IActorExceptionHandler, public NTable::IScan {
protected:
    const ui64 TabletId = 0;
    const ui64 BuildId = 0;
    const TAutoPtr<TEvDataShard::TEvEncodePqResponse> Response;
    const TActorId ResponseActorId;

    std::optional<NTableIndex::NIvfPq::TClusterId> Parent;

    TTags ScanTags;
    NTable::TPos EmbeddingPos = 0;
    NTable::TPos DataPos = 1;

    ui64 ReadRows = 0;
    ui64 ReadBytes = 0;

    TBatchRowsUploader Uploader;

    TBufferData* OutputBuf = nullptr;

    const TIndexBuildScanSettings ScanSettings;

    TSerializedCellVec LastAckedKey;
    TSerializedCellVec PendingCheckpointKey;
    ui64 NextCheckpointAtBytes = 0;

    bool InForeign = false;
    NTable::TPos IsForeignPos = 0;

    bool IsExhausted = false;

    IDriver* Driver = nullptr;
    NYql::TIssues Issues;

    TLead Lead;

    std::unique_ptr<TProductQuantizer> ProductQuantizer;
    const ui64 M;
    const ui32 SubspaceBits;

public:
    static constexpr NKikimrServices::TActivity::EType ActorActivityType() {
        return NKikimrServices::TActivity::ENCODE_PQ_SCAN_ACTOR;
    }

    TEncodePqScan(ui64 tabletId, const TUserTable& table, const NKikimrTxDataShard::TEvEncodePqRequest& request,
        const TActorId& responseActorId, TAutoPtr<TEvDataShard::TEvEncodePqResponse>&& response,
        TLead&& lead, std::unique_ptr<TProductQuantizer>&& productQuantizer)
        : TActor(&TThis::StateWork)
        , TabletId(tabletId)
        , BuildId(request.GetId())
        , Response(std::move(response))
        , ResponseActorId(responseActorId)
        , Parent(request.HasParent() ? std::make_optional(request.GetParent()) : std::nullopt)
        , Uploader(request.GetDatabaseName(), request.GetScanSettings())
        , ScanSettings(request.GetScanSettings())
        , Lead(std::move(lead))
        , ProductQuantizer(std::move(productQuantizer))
        , M(ProductQuantizer->SubspaceCount)
        , SubspaceBits(request.GetSubspaceBits())
    {
        LOG_I("Create " << Debug());
        NextCheckpointAtBytes = ScanSettings.GetMaxCheckpointBytes();

        // InForeign = request.GetSkipOverlapForeign();

        const auto& embedding = request.GetEmbeddingColumn();
        const auto& data = request.GetDataColumns();
        ScanTags = NKMeans::MakeScanTags(
            table, embedding, data,
            false, EmbeddingPos, DataPos, InForeign ? &IsForeignPos : nullptr);
        Lead.SetTags(ScanTags);

        OutputBuf = Uploader.AddDestination(request.GetOutputName(), MakePqOutputTypes(table, data, Parent.has_value()));
    }

    TInitialState Prepare(IDriver* driver, TIntrusiveConstPtr<TScheme>) final {
        TActivationContext::AsActorContext().RegisterWithSameMailbox(this);
        LOG_I("Prepare " << Debug());

        Driver = driver;
        Uploader.SetOwner(SelfId());

        return {EScan::Feed, {}};
    }

    TAutoPtr<IDestructable> Finish(const std::exception& exc) final {
        Uploader.AddIssue(exc);
        return Finish(EStatus::Exception);
    }

    TAutoPtr<IDestructable> Finish(EStatus status) final {
        auto& record = Response->Record;
        record.MutableMeteringStats()->SetReadRows(ReadRows);
        record.MutableMeteringStats()->SetReadBytes(ReadBytes);
        record.MutableMeteringStats()->SetCpuTimeUs(Driver->GetTotalCpuTimeUs());

        if (LastAckedKey.GetBuffer()) {
            record.SetLastKeyAck(LastAckedKey.GetBuffer());
        }

        Uploader.Finish(record, status);

        if (Response->Record.GetStatus() == NKikimrIndexBuilder::DONE) {
            LOG_N("Done " << Debug() << " " << Response->Record.ShortDebugString());
        } else {
            LOG_E("Failed " << Debug() << " " << Response->Record.ShortDebugString());
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

    EScan PageFault() final {
        LOG_T("PageFault " << Debug());
        return EScan::Feed;
    }

    EScan Seek(TLead& lead, ui64 seq) final {
        LOG_T("Seek " << seq << " " << Debug());

        if (IsExhausted) {
            return Uploader.CanFinish()
                ? EScan::Final
                : EScan::Sleep;
        }

        lead = Lead;

        return EScan::Feed;
    }

    EScan Feed(TArrayRef<const TCell> key, const TRow& row) final {
        // LOG_T("Feed " << Debug());

        ++ReadRows;
        ReadBytes += CountRowCellBytes(key, *row);

        Feed(key, *row);

        return Uploader.ShouldWaitUpload() ? EScan::Sleep : EScan::Feed;
    }

    EScan Exhausted() final {
        LOG_T("Exhausted " << Debug());

        IsExhausted = true;

        // call Seek to wait uploads
        return EScan::Reset;
    }

protected:
    STFUNC(StateWork) {
        switch (ev->GetTypeRewrite()) {
            HFunc(TEvTxUserProxy::TEvUploadRowsResponse, Handle);
            CFunc(TEvents::TSystem::Wakeup, HandleWakeup);
            default:
                LOG_E("StateWork unexpected event type: " << ev->GetTypeRewrite()
                    << " event: " << ev->ToString() << " " << Debug());
        }
    }

    void HandleWakeup(const NActors::TActorContext&)
    {
        LOG_D("Retry upload " << Debug());

        Uploader.RetryUpload();
    }

    void Handle(TEvTxUserProxy::TEvUploadRowsResponse::TPtr& ev, const TActorContext& ctx)
    {
        LOG_D("Handle TEvUploadRowsResponse " << Debug()
            << " ev->Sender: " << ev->Sender.ToString());

        if (!Driver) {
            return;
        }

        bool batchUploaded = Uploader.Handle(ev);

        if (Uploader.GetUploadStatus().IsSuccess()) {
            if (batchUploaded && PendingCheckpointKey.GetBuffer() && Uploader.AllFlushed()
                && Uploader.GetUploadBytes() >= NextCheckpointAtBytes) {
                NextCheckpointAtBytes = Uploader.GetUploadBytes() + ScanSettings.GetMaxCheckpointBytes();
                LastAckedKey = PendingCheckpointKey;
                PendingCheckpointKey = {};

                auto progress = MakeHolder<TEvDataShard::TEvEncodePqResponse>();
                auto& rec = progress->Record;
                rec.SetId(BuildId);
                rec.SetTabletId(TabletId);
                rec.SetRequestSeqNoGeneration(Response->Record.GetRequestSeqNoGeneration());
                rec.SetRequestSeqNoRound(Response->Record.GetRequestSeqNoRound());
                rec.SetStatus(NKikimrIndexBuilder::EBuildStatus::IN_PROGRESS);
                rec.SetLastKeyAck(LastAckedKey.GetBuffer());
                Send(ResponseActorId, progress.Release());
            }
            Driver->Touch(EScan::Feed);
            return;
        }

        if (auto retryAfter = Uploader.GetRetryAfter(); retryAfter) {
            LOG_N("Got retriable error, " << Debug() << " " << Uploader.GetUploadStatus().ToString());
            ctx.Schedule(*retryAfter, new TEvents::TEvWakeup());
            return;
        }

        LOG_N("Got error, abort scan, " << Debug() << " " << Uploader.GetUploadStatus().ToString());

        Driver->Touch(EScan::Final);
    }

    TString Debug() const {
        return TStringBuilder{}
            << "TEncodePqScan TabletId: " << TabletId
            << " Id: " << BuildId
            << " ProductQuantizer: " << ProductQuantizer->Debug();
    }

    void Feed(TArrayRef<const TCell> key, TArrayRef<const TCell> row) {
        // if (InForeign) {
        //     bool foreign = row.at(IsForeignPos).AsValue<bool>();
        //     if (foreign) {
        //         // Skip rows from "non-domestic" clusters to not affect K-means centroids
        //         return;
        //     }
        // }

        const auto dataColumns = row.Slice(DataPos);

        const auto embedding = row.at(EmbeddingPos).AsBuf();
        if (!ProductQuantizer->IsValidEmbedding(embedding)) {
            return;
        }

        const auto code = ProductQuantizer->Quantize(embedding);
        const TString serializedCode = NKikimr::NIvfPq::NPackedNBitVector::Serialize(code, SubspaceBits);

        TVector<TCell> data(::Reserve(dataColumns.size() + 1));
        data.push_back(TCell{serializedCode});
        data.insert(data.end(), dataColumns.begin(), dataColumns.end());

        OutputBuf->AddRow(key, data, key);
    }
};

class TDataShard::TTxHandleSafeEncodePqScan final: public NTabletFlatExecutor::TTransactionBase<TDataShard> {
public:
    TTxHandleSafeEncodePqScan(TDataShard* self, TEvDataShard::TEvEncodePqRequest::TPtr&& ev)
        : TTransactionBase(self)
        , Ev(std::move(ev))
    {
    }

    bool Execute(TTransactionContext&, const TActorContext& ctx) final
    {
        Self->HandleSafe(Ev, ctx);
        return true;
    }

    void Complete(const TActorContext&) final
    {
    }

private:
    TEvDataShard::TEvEncodePqRequest::TPtr Ev;
};

void TDataShard::Handle(TEvDataShard::TEvEncodePqRequest::TPtr& ev, const TActorContext&)
{
    Execute(new TTxHandleSafeEncodePqScan(this, std::move(ev)));
}

void TDataShard::HandleSafe(TEvDataShard::TEvEncodePqRequest::TPtr& ev, const TActorContext& ctx) {
    auto& request = ev->Get()->Record;
    const ui64 id = request.GetId();
    auto rowVersion = request.HasSnapshotStep() || request.HasSnapshotTxId()
        ? TRowVersion(request.GetSnapshotStep(), request.GetSnapshotTxId())
        : GetMvccTxVersion(EMvccTxMode::ReadOnly);
    TScanRecord::TSeqNo seqNo = {request.GetSeqNoGeneration(), request.GetSeqNoRound()};

    try {
        auto response = MakeHolder<TEvDataShard::TEvEncodePqResponse>();
        FillScanResponseCommonFields(*response, id, TabletID(), seqNo);

        LOG_N("Starting TEncodePqScan TabletId: " << TabletID()
            << " " << request.ShortDebugString()
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
                LOG_E("Rejecting TEncodePqScan bad request TabletId: " << TabletID()
                    << " " << request.ShortDebugString()
                    << " with response " << response->Record.ShortDebugString());
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

        {
            TVector<TCell> from = {TCell()};
            TVector<TCell> to = {};

            if (request.HasParent()) {
                const auto parent = request.GetParent();
                from = {TCell::Make(parent), TCell()};
                to = {TCell::Make(parent)};
            }

            const TTableRange requestedRange = TTableRange(from, true, to, true);
            TTableRange scanRange = Intersect(userTable.KeyColumnTypes, requestedRange, userTable.Range.ToTableRange());

            if (scanRange.IsEmptyRange(userTable.KeyColumnTypes)) {
                badRequest(TStringBuilder() << "Requested range doesn't intersect with table range:"
                    << " requestedRange: " << DebugPrintRange(userTable.KeyColumnTypes, requestedRange, *AppData()->TypeRegistry)
                    << " tableRange: " << DebugPrintRange(userTable.KeyColumnTypes, userTable.Range.ToTableRange(), *AppData()->TypeRegistry)
                    << " scanRange: " << DebugPrintRange(userTable.KeyColumnTypes, scanRange, *AppData()->TypeRegistry));
            }

            if (request.HasKeyRange()) {
                TSerializedTableRange resumeRange;
                resumeRange.Load(request.GetKeyRange());
                scanRange = Intersect(userTable.KeyColumnTypes, resumeRange.ToTableRange(), scanRange);
                if (scanRange.IsEmptyRange(userTable.KeyColumnTypes)) {
                    badRequest(TStringBuilder() << "Requested resume range doesn't intersect with scan range:"
                        << " resumeRange: " << DebugPrintRange(userTable.KeyColumnTypes, resumeRange.ToTableRange(), *AppData()->TypeRegistry)
                        << " scanRange: " << DebugPrintRange(userTable.KeyColumnTypes, scanRange, *AppData()->TypeRegistry));
                }
            }

            FillLeadFromRange(scanRange, lead);
        }

        auto tags = GetAllTags(userTable);
        if (!tags.contains(request.GetEmbeddingColumn())) {
            badRequest(TStringBuilder() << "Unknown embedding column: " << request.GetEmbeddingColumn());
        }

        // 3. Validating vector index settings
        TString error;
        auto productQuantizer = TProductQuantizer::Create(request.GetSubspaces(), request.GetSettings(), 0, error);
        if (!productQuantizer) {
            badRequest(error);
        } else if (request.SubquantizersSize() != request.GetSubspaces()) {
            badRequest(TStringBuilder() << "Invalid subquantizers count: " << request.SubquantizersSize() << " expected " << request.GetSubspaces());
        } else {
            for (size_t i = 0; i < request.GetSubspaces(); ++i) {
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
        
        TAutoPtr<NTable::IScan> scan = new TEncodePqScan(
            TabletID(), userTable, request, ev->Sender, std::move(response),
            std::move(lead), std::move(productQuantizer)
        );

        StartScan(this, std::move(scan), id, seqNo, rowVersion, userTable.LocalTid);
    } catch (const std::exception& exc) {
        FailScan<TEvDataShard::TEvEncodePqResponse>(id, TabletID(), ev->Sender, seqNo, exc, "TEncodePqScan");
    }
}

}