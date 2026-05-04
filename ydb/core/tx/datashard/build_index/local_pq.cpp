#include "common_helper.h"
#include "kmeans_helper.h"

#include <ydb/core/tx/datashard/datashard_impl.h>

#include <ydb/core/base/kmeans_clusters.h>
#include <ydb/core/tablet_flat/flat_scan_iface.h>

#include <ydb/library/actors/core/actor.h>
#include <ydb/library/yql/udfs/common/knn/knn-serializer-shared.h>

#include <util/generic/vector.h>

#include <memory>

namespace NKikimr::NDataShard {
using namespace NKMeans;

namespace {
    template <typename T>
    TVector<TString> SplitEmbedding(const TArrayRef<const char> embedding, const ui32 m) {
        using NKnnVectorSerialization::TDeserializer;
        using NKnnVectorSerialization::TSerializer;

        TDeserializer<T> deserializer(TStringBuf(embedding.data(), embedding.size()));

        Y_ENSURE(deserializer.GetElementCount() % m == 0);
        const size_t segmentSize = deserializer.GetElementCount() / m;
        TVector<TString> segments;

        TStringBuilder builder;
        TSerializer<T> serializer(&builder.Out);

        size_t i = 0;
        deserializer.DoDeserialize([&](const T& element) {
            serializer.HandleElement(element);
            ++i;

            if (i == segmentSize) {
                serializer.Finish();
                segments.emplace_back();
                builder.Out.Flush();
                builder.swap(segments.back());

                i = 0;
                serializer = std::move(TSerializer<T>(&builder.Out));
            }
        });

        return segments;
    }

    std::shared_ptr<NTxProxy::TUploadTypes> MakeCodebookTypes() {
        auto result = std::make_shared<NTxProxy::TUploadTypes>();
        
        Ydb::Type type;
        type.set_type_id(NTableIndex::NIvfPq::ClusterIdType);
        result->emplace_back(NTableIndex::NIvfPq::ParentColumn, type);

        type.set_type_id(NTableIndex::NIvfPq::SegmentIdxType);
        result->emplace_back(NTableIndex::NIvfPq::SegmentColumn, type);

        type.set_type_id(NTableIndex::NIvfPq::CodeType);
        result->emplace_back(NTableIndex::NIvfPq::CodeColumn, type);

        type.set_type_id(Ydb::Type::STRING);
        result->emplace_back(NTableIndex::NIvfPq::CentroidColumn, type);

        return result;
    }
}



class TLocalPqScan : public TActor<TLocalPqScan>, public IActorExceptionHandler, public NTable::IScan {
    using EState = NKikimrTxDataShard::EKMeansState;

    NTableIndex::NKMeans::TClusterId Parent = 0;
    NTableIndex::NKMeans::TClusterId Child = 0;

    EState State;
    const EState UploadState = NKikimrTxDataShard::UPLOAD_BUILD_TO_POSTING; // TODO(raydzast): definitely not this

    IDriver* Driver = nullptr;

    TLead Lead;

    ui64 TabletId = 0;
    ui64 BuildId = 0;

    ui64 ReadRows = 0;
    ui64 ReadBytes = 0;

    TBatchRowsUploader Uploader;

    TBufferData* CodebookBuf = nullptr;
    TBufferData* OutputBuf = nullptr;
    // TBufferData* UploadBuf = nullptr;

    const ui32 Dimensions;
    const ui32 K;
    const ui32 M;
    NTable::TPos EmbeddingPos = 0;
    NTable::TPos DataPos = 1;
    bool IsEmpty = false;
    // const ui32 OverlapClusters = 0;
    // const double OverlapRatio = 0;
    bool OutForeign = false;
    bool InForeign = false;
    NTable::TPos IsForeignPos = 0;

    const TIndexBuildScanSettings ScanSettings;

    TTags ScanTags;

    TUploadStatus UploadStatus;

    // ui64 UploadRows = 0;
    // ui64 UploadBytes = 0;

    TActorId ResponseActorId;
    TAutoPtr<TEvDataShard::TEvLocalPqResponse> Response;

    // FIXME: save PrefixRows as std::vector<std::pair<TSerializedCellVec, TSerializedCellVec>> to avoid parsing
    const ui32 PrefixColumns;
    TSerializedCellVec Prefix;
    TBufferData PrefixRows;
    bool IsFirstPrefixFeed = true;
    bool IsPrefixRowsValid = true;

    bool IsExhausted = false;

    TVector<NScheme::TTypeInfo> KeyTypes;
    TSerializedCellVec LastAckedKey;
    TSerializedCellVec PendingCheckpointKey;
    ui64 NextCheckpointAtBytes = 0;

    NKMeans::TSampler Sampler;

    std::vector<std::pair<ui32, double>> TmpClusters;

    const TVector<std::unique_ptr<NKikimr::NKMeans::IClusters>> ClustersBySubspace;

public:
    static constexpr NKikimrServices::TActivity::EType ActorActivityType()
    {
        return NKikimrServices::TActivity::LOCAL_PQ_SCAN_ACTOR;
    }

    TLocalPqScan(ui64 tabletId, const TUserTable& table, const NKikimrTxDataShard::TEvLocalPqRequest& request,
        const TActorId& responseActorId, TAutoPtr<TEvDataShard::TEvLocalPqResponse>&& response,
        TLead&& lead, TVector<std::unique_ptr<NKikimr::NKMeans::IClusters>>&& clustersBySubspace)
        : TActor{&TThis::StateWork}
        , Parent{request.GetParentFrom()}
        // , Child{request.GetChild()}
        , State{EState::SAMPLE}
        // , UploadState{request.GetUpload()}
        , Lead{std::move(lead)}
        , TabletId(tabletId)
        , BuildId{request.GetId()}
        , Uploader(request.GetDatabaseName(), request.GetScanSettings())
        , Dimensions(request.GetSettings().vector_dimension())
        , K(1 << request.GetNBits())
        , M(request.GetM())
        // , OverlapClusters(request.GetOverlapClusters() ? request.GetOverlapClusters() : 1)
        // , OverlapRatio(request.GetOverlapRatio())
        , ScanSettings(request.GetScanSettings())
        , ResponseActorId{responseActorId}
        , Response{std::move(response)}
        , PrefixColumns{request.GetParentFrom() == 0 && request.GetParentTo() == 0 ? 0u : 1u}
        , KeyTypes(table.KeyColumnTypes)
        , Sampler(K, request.GetSeed())
        , ClustersBySubspace(std::move(clustersBySubspace))
    {
        LOG_I("Create " << Debug());
        NextCheckpointAtBytes = ScanSettings.GetMaxCheckpointBytes();

        // InForeign = OverlapClusters > 1 && (request.GetUpload() == NKikimrTxDataShard::UPLOAD_BUILD_TO_BUILD
        //     || request.GetUpload() == NKikimrTxDataShard::UPLOAD_BUILD_TO_POSTING);
        // OutForeign = OverlapClusters > 1 && request.GetOverlapOutForeign();

        const auto& embedding = request.GetEmbeddingColumn();
        const auto& data = request.GetDataColumns();
        ScanTags = MakeScanTags(table, embedding, data, true,
            EmbeddingPos, DataPos, InForeign ? &IsForeignPos : nullptr);
        Lead.SetTags(ScanTags);

        auto codebookTypes = MakeCodebookTypes();
        CodebookBuf = Uploader.AddDestination(request.GetCodebookName(), std::move(codebookTypes));

        auto outputTypes = MakeOutputTypes(table, UploadState, embedding, data, {}, OutForeign);
        OutputBuf = Uploader.AddDestination(request.GetOutputName(), std::move(outputTypes));
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

    TAutoPtr<IDestructable> Finish(const EStatus status) final {
        auto& record = Response->Record;
        record.MutableMeteringStats()->SetReadRows(ReadRows);
        record.MutableMeteringStats()->SetReadBytes(ReadBytes);
        record.MutableMeteringStats()->SetCpuTimeUs(Driver->GetTotalCpuTimeUs());
        // TODO(raydzast): why IsEmpty is set for whole datashard even if only one prefix was empty?
        record.SetIsEmpty(IsEmpty);

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

        LOG_I("SEEK " << seq << " LEAD " << Lead.Key.GetCells()[0].AsValue<ui64>());

        // TODO(raydzast): not obvious what the fuck happening
        if (IsExhausted) {
            return Uploader.CanFinish()
                ? EScan::Final
                : EScan::Sleep;
        }

        lead = Lead;

        return EScan::Feed;
    }

    EScan Feed(const TArrayRef<const TCell> key, const TRow& row) final {
        // LOG_T("Feed " << Debug());

        ++ReadRows;
        ReadBytes += CountRowCellBytes(key, *row);

        if (PrefixColumns && Prefix && !TCellVectorsEquals{}(Prefix.GetCells(), key.subspan(0, PrefixColumns))) {
            if (!FinishPrefix()) {
                // scan current prefix rows with a new state again
                return EScan::Reset;
            }
        }

        if (PrefixColumns && !Prefix) {
            Prefix = TSerializedCellVec{key.subspan(0, PrefixColumns)};
            auto newParent = key.at(0).template AsValue<ui64>();
            Child += (newParent - Parent) * K;
            Parent = newParent;
        }

        if (IsFirstPrefixFeed && IsPrefixRowsValid) {
            PrefixRows.AddRow(key, *row);
            if (PrefixRows.HasReachedLimits(ScanSettings)) {
                PrefixRows.Clear();
                IsPrefixRowsValid = false;
            }
        }

        Feed(key, *row);

        return Uploader.ShouldWaitUpload() ? EScan::Sleep : EScan::Feed;
    }

    EScan Exhausted() final {
        LOG_T("Exhausted " << Debug());

        if (!FinishPrefix()) {
            return EScan::Reset;
        }

        IsExhausted = true;

        // call Seek to wait uploads
        return EScan::Reset;
    }

protected:
    STFUNC(StateWork)
    {
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
                NextCheckpointAtBytes = Uploader.GetUploadBytes() + ScanSettings.GetMaxCheckpointBytes();;
                LastAckedKey = PendingCheckpointKey;
                PendingCheckpointKey = {};

                auto progress = MakeHolder<TEvDataShard::TEvLocalPqResponse>();
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

    void StartNewPrefix()
    {
        State = EState::SAMPLE;
        Lead.Valid = true;
        Lead.Key = TSerializedCellVec(Prefix.GetCells()); // seek to (prefix, inf)
        Lead.Relation = NTable::ESeek::Upper;
        Prefix = {};
        IsFirstPrefixFeed = true;
        IsPrefixRowsValid = true;
        PrefixRows.Clear();
        Sampler.Finish();

        for (const auto& c : ClustersBySubspace) {
            c->Clear();
        }
    }

    bool FinishPrefix()
    {
        if (FinishPrefixImpl()) {
            PendingCheckpointKey = Prefix;
            StartNewPrefix();
            LOG_T("FinishPrefix finished " << Debug());
            return true;
        } else {
            IsFirstPrefixFeed = false;

            if (IsPrefixRowsValid) {
                // LOG_T("FinishPrefix not finished, manually feeding " << PrefixRows.GetRows() << " saved rows " << Debug());
                for (ui64 iteration = 0; ; iteration++) {
                    LOG_T("FinishPrefix not finished, iteration " << iteration << " manually feeding " << PrefixRows.GetRows() << " saved rows " << Debug());
                    for (const auto& [key, row_] : *PrefixRows.GetRowsData()) {
                        TSerializedCellVec row(row_);
                        Feed(key.GetCells(), row.GetCells());
                    }
                    if (FinishPrefixImpl()) {
                        PendingCheckpointKey = Prefix;
                        StartNewPrefix();
                        LOG_T("FinishPrefix finished in " << iteration << " iterations " << Debug());
                        return true;
                    } else {
                        LOG_T("FinishPrefix not finished in " << iteration << " iterations " << Debug());
                    }
                }
            } else {
                LOG_T("FinishPrefix not finished, rescanning rows " << Debug());
            }

            return false;
        }
    }

    bool FinishPrefixImpl()
    {
        if (State == EState::SAMPLE) {
            State = EState::KMEANS;
            auto rows = Sampler.Finish().second;
            if (rows.size() == 0) {
                // We don't need to do anything,
                // because this datashard doesn't have valid embeddings for this prefix
                IsEmpty = true;
                return true;
            }
            if (rows.size() < K) {
                // if this datashard has less than K valid embeddings for this parent
                // lets make single centroid for it
                rows.resize(1);
            }

            LOG_T("FinishPrefixImpl initializing ClustersBySubspace");

            TVector<TVector<TString>> segmentsBySubspace(M, TVector<TString>(rows.size()));
            for (size_t rowIdx = 0; rowIdx < rows.size(); ++rowIdx) {
                const auto embedding = rows.at(rowIdx);
                // TODO(raydzast): remove hardcoded type
                auto subspaces = SplitEmbedding<ui8>(embedding, M);
                for (size_t i = 0; i < M; ++i) {
                    segmentsBySubspace[i][rowIdx] = std::move(subspaces[i]);
                }
            }

            for (size_t i = 0; i < M; ++i) {
                bool ok = ClustersBySubspace[i]->SetClusters(std::move(segmentsBySubspace[i]));
                Y_ENSURE(ok);
                ClustersBySubspace[i]->SetRound(1);
            }

            LOG_T("FinishPrefixImpl initialized ClustersBySubspace: " << Debug());
            return false; // do KMEANS
        }

        if (State == EState::KMEANS) {
            // TODO(raydzast): keep track of finished KMeans to not recompute unnecessarily
            bool finished = true;
            for (size_t i = 0; i < M; ++i) {
                finished |= ClustersBySubspace[i]->NextRound();    
            }

            if (finished) {
                FormCodebookRows();
                State = UploadState; // do UPLOAD_*
            }
            return false;
        }

        if (State == UploadState) {
            return true;
        }

        Y_ENSURE(false);
        return true;
    }

    void Feed(const TArrayRef<const TCell> key, const TArrayRef<const TCell> row)
    {
        switch (State) {
            case EState::SAMPLE:
                FeedSample(row);
                break;
            case EState::KMEANS:
                FeedKMeans(row);
                break;
            case EState::UPLOAD_MAIN_TO_POSTING:
                FeedMainToPosting(key, row);
                break;
            case EState::UPLOAD_BUILD_TO_POSTING:
                FeedBuildToPosting(key, row);
                break;
            // case EState::UPLOAD_MAIN_TO_BUILD:
                // FeedMainToBuild(key, row);
                // break;
            // case EState::UPLOAD_BUILD_TO_BUILD:
                // FeedBuildToBuild(key, row);
                // break;
            default:
                Y_ENSURE(false);
        }
    }

    void FeedSample(TArrayRef<const TCell> row)
    {
        // TODO(raydzast): support for foreigns
        // if (InForeign) {
        //     bool foreign = row.at(IsForeignPos).AsValue<bool>();
        //     if (foreign) {
        //         // Skip rows from "non-domestic" clusters to not affect K-means centroids
        //         return;
        //     }
        // }

        const auto embedding = row.at(EmbeddingPos).AsRef();
        // TODO(raydzast): remove hardcoded type
        const auto subspaces = SplitEmbedding<ui8>(embedding, M);
        for (size_t i = 0; i < M; ++i) {
            const auto& subEmbedding = subspaces[i];
            if (!ClustersBySubspace[i]->IsExpectedFormat(subEmbedding)) {
                return;
            }
        }
        // if (!Clusters->IsExpectedFormat(embedding)) {
        //     return;
        // }

        Sampler.Add([&embedding](){
            return TString(embedding.data(), embedding.size());
        });
    }

    void FeedKMeans(TArrayRef<const TCell> row)
    {
        if (InForeign) {
            bool foreign = row.at(IsForeignPos).AsValue<bool>();
            if (foreign) {
                // Skip rows from "non-domestic" clusters to not affect K-means centroids
                return;
            }
        }

        const auto embedding = row.at(EmbeddingPos).AsRef();
        // TODO(raydzast): remove hardcoded type
        const auto subspaces = SplitEmbedding<ui8>(embedding, M);
        for (size_t i = 0; i < M; ++i) {
            const auto& subEmbedding = subspaces[i];
            if (auto pos = ClustersBySubspace[i]->FindCluster(subEmbedding); pos) {
                ClustersBySubspace[i]->AggregateToCluster(*pos, subEmbedding);
            }
        }

        // if (auto pos = Clusters->FindCluster(row, EmbeddingPos); pos) {
        //     Clusters->AggregateToCluster(*pos, row.at(EmbeddingPos).AsRef());
        // }
    }

    void FeedFinal(TArrayRef<const TCell> row, TArrayRef<const TCell> sourcePk,
        TArrayRef<const TCell> dataColumns, TArrayRef<const TCell> origKey, bool isPostingLevel)
    {
        // Clusters->FindClusters(row.at(EmbeddingPos).AsBuf(), TmpClusters, OverlapClusters, OverlapRatio);
        if (OutForeign) {
            bool foreign = false;
            if (InForeign) {
                foreign = row.at(IsForeignPos).AsValue<bool>();
            }
            for (auto& [pos, distance]: TmpClusters) {
                AddRowToDataWithForeign(*OutputBuf, Child + pos, sourcePk, dataColumns, origKey, foreign, distance, isPostingLevel);
                foreign = true;
            }
        } else {
            for (auto& [pos, _]: TmpClusters) {
                AddRowToData(*OutputBuf, Child + pos, sourcePk, dataColumns, origKey, isPostingLevel);
            }
        }
    }

    void FeedMainToBuild(TArrayRef<const TCell> key, TArrayRef<const TCell> row)
    {
        FeedFinal(row, key, row.Slice(DataPos), key, false);
    }

    void FeedMainToPosting(TArrayRef<const TCell> key, TArrayRef<const TCell> row)
    {
        FeedFinal(row, key, row.Slice(DataPos), key, true);
    }

    void FeedBuildToBuild(TArrayRef<const TCell> key, TArrayRef<const TCell> row)
    {
        FeedFinal(row, key.Slice(1), row.Slice(DataPos), key, false);
    }

    void FeedBuildToPosting(TArrayRef<const TCell> key, TArrayRef<const TCell> row)
    {
        FeedFinal(row, key.Slice(1), row.Slice(DataPos), key, true);
    }
    
    void FormCodebookRows() {
        for (size_t i = 0; i < M; ++i) {
            const auto& clusters = ClustersBySubspace[i]->GetClusters();

            for (size_t code = 0; code < clusters.size(); ++code) {
                const TString& centroid = clusters[code];

                std::array<TCell, 3> pk;
                pk[0] = TCell::Make<NIvfPq::TClusterId>(Parent);
                pk[1] = TCell::Make<NIvfPq::TSegmentIdx>(i);
                pk[2] = TCell::Make<NIvfPq::TCode>(code);

                std::array<TCell, 1> data;
                data[0] = TCell{centroid};

                CodebookBuf->AddRow(pk, data);
            }
        }
    }

    TString Debug() const
    {
        TStringBuilder log;
        log << "TLocalPqScan TabletId: " << TabletId << " Id: " << BuildId
            << " State: " << State
            << " Parent: " << Parent // << " Child: " << Child
            << " " << Sampler.Debug();

        for (size_t i = 0; i < M; ++i) {
            log << " subspace: " << i << " " << ClustersBySubspace[i]->Debug();
        }
        log << " " << Uploader.Debug();

        return log;
    }
};

class TDataShard::TTxHandleSafeLocalPqScan final: public NTabletFlatExecutor::TTransactionBase<TDataShard> {
public:
    TTxHandleSafeLocalPqScan(TDataShard* self, TEvDataShard::TEvLocalPqRequest::TPtr&& ev)
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
    TEvDataShard::TEvLocalPqRequest::TPtr Ev;
};

void TDataShard::Handle(TEvDataShard::TEvLocalPqRequest::TPtr& ev, const TActorContext&)
{
    Execute(new TTxHandleSafeLocalPqScan(this, std::move(ev)));
}

void TDataShard::HandleSafe(TEvDataShard::TEvLocalPqRequest::TPtr& ev, const TActorContext& ctx) {
    auto& request = ev->Get()->Record;
    const ui64 id = request.GetId();
    auto rowVersion = request.HasSnapshotStep() || request.HasSnapshotTxId()
        ? TRowVersion(request.GetSnapshotStep(), request.GetSnapshotTxId())
        : GetMvccTxVersion(EMvccTxMode::ReadOnly);
    TScanRecord::TSeqNo seqNo = {request.GetSeqNoGeneration(), request.GetSeqNoRound()};

    try {
        auto response = MakeHolder<TEvDataShard::TEvLocalPqResponse>();
        FillScanResponseCommonFields(*response, id, TabletID(), seqNo);

        LOG_N("Starting TLocalPqScan TabletId: " << TabletID()
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
                LOG_E("Rejecting TLocalPqScan bad request TabletId: " << TabletID()
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

        const auto parentFrom = request.GetParentFrom();
        const auto parentTo = request.GetParentTo();
        NTable::TLead lead;
        if (parentFrom == 0) {
            // TODO(raydzast): need to be understood
            // if (request.HasKeyRange()) {
            //     TSerializedTableRange resumeRange;
            //     resumeRange.Load(request.GetKeyRange());
            //     auto scanRange = Intersect(userTable.KeyColumnTypes, resumeRange.ToTableRange(), userTable.Range.ToTableRange());
            //     lead = CreateLeadFrom(scanRange);
            // } else {
                lead.To({}, NTable::ESeek::Lower);
            // }
        } else if (parentFrom > parentTo) {
            badRequest(TStringBuilder() << "Parent from " << parentFrom << " should be less or equal to parent to " << parentTo);
        } else {
            TCell from = TCell::Make(parentFrom - 1);
            TCell to = TCell::Make(parentTo);
            TTableRange parentRange{{&from, 1}, false, {&to, 1}, true};
            auto scanRange = Intersect(userTable.KeyColumnTypes, parentRange, userTable.Range.ToTableRange());
            if (scanRange.IsEmptyRange(userTable.KeyColumnTypes)) {
                badRequest(TStringBuilder() << "Requested range doesn't intersect with table range:"
                    << " requestedRange: " << DebugPrintRange(userTable.KeyColumnTypes, parentRange, *AppData()->TypeRegistry)
                    << " tableRange: " << DebugPrintRange(userTable.KeyColumnTypes, userTable.Range.ToTableRange(), *AppData()->TypeRegistry)
                    << " scanRange: " << DebugPrintRange(userTable.KeyColumnTypes, scanRange, *AppData()->TypeRegistry));
            }
            // TODO(raydzast): need to be understood
            // if (request.HasKeyRange()) {
            //     TSerializedTableRange resumeRange;
            //     resumeRange.Load(request.GetKeyRange());
            //     auto resumeScanRange = Intersect(userTable.KeyColumnTypes, resumeRange.ToTableRange(), scanRange);
            //     lead = CreateLeadFrom(resumeScanRange);
            // } else {
                lead.To(parentRange.From, NTable::ESeek::Upper);
                lead.Until(parentRange.To, true);
            // }
        }

        TVector<std::unique_ptr<NKikimr::NKMeans::IClusters>> clustersBySubspace;
        for (size_t i = 0; i < request.GetM(); ++i) {
            TString error;
            auto settings = request.GetSettings();
            settings.set_vector_dimension(settings.vector_dimension() / request.GetM());
            auto clusters = NKikimr::NKMeans::CreateClusters(std::move(settings), request.GetKMeansRounds(), error);
            if (!clusters) {
                badRequest(error);
                auto sent = trySendBadRequest();
                Y_ENSURE(sent);
                return;
            }
            clustersBySubspace.push_back(std::move(clusters));
        }
        
        TAutoPtr<NTable::IScan> scan = new TLocalPqScan(
            TabletID(), userTable, request, ev->Sender, std::move(response),
            std::move(lead), std::move(clustersBySubspace)
        );

        StartScan(this, std::move(scan), id, seqNo, rowVersion, userTable.LocalTid);
    } catch (const std::exception& exc) {
        FailScan<TEvDataShard::TEvLocalPqResponse>(id, TabletID(), ev->Sender, seqNo, exc, "TLocalPqScan");
    }
}

}