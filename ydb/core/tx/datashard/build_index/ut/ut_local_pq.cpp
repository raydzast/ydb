#include "ut_helpers.h"

#include <library/cpp/testing/unittest/registar.h>
#include <ydb/core/testlib/test_client.h>

#include <atomic>

namespace NKikimr {
using namespace Tests;
using Ydb::Table::VectorIndexSettings;

static std::atomic<ui64> sId = 1;
static const TString kDatabaseName = "/Root";
static const TString kBuildTable = "/Root/table-build";
static const TString kCodebookTable = "/Root/table-codebook";
static const TString kPostingTable = "/Root/table-posting";

Y_UNIT_TEST_SUITE(TTxDataShardLocalPqScan) {

    static std::tuple<TString, TString> DoLocalPq(
        Tests::TServer::TPtr server, TActorId sender,
        NTableIndex::NKMeans::TClusterId parentFrom,
        NTableIndex::NKMeans::TClusterId parentTo,
        NKikimrTxDataShard::EKMeansState,
        ui64 seed, ui64 m, ui64 nbits,
        VectorIndexSettings::VectorType type, VectorIndexSettings::Metric metric,
        ui32 maxBatchRows = 50000, [[maybe_unused]] ui32 overlapClusters = 0, bool expectEmpty = false,
        [[maybe_unused]] std::optional<TSerializedTableRange> keyRange = {})
    {
        auto id = sId.fetch_add(1, std::memory_order_relaxed);
        auto& runtime = *server->GetRuntime();
        auto snapshot = CreateVolatileSnapshot(server, {kBuildTable});
        auto datashards = GetTableShards(server, sender, kBuildTable);
        TTableId tableId = ResolveTableId(server, sender, kBuildTable);

        TString err;

        for (auto tid : datashards) {
            auto ev1 = std::make_unique<TEvDataShard::TEvLocalPqRequest>();
            auto ev2 = std::make_unique<TEvDataShard::TEvLocalPqRequest>();
            auto fill = [&](std::unique_ptr<TEvDataShard::TEvLocalPqRequest>& ev) {
                auto& rec = ev->Record;
                rec.SetId(1);

                rec.SetSeqNoGeneration(id);
                rec.SetSeqNoRound(1);

                rec.SetTabletId(tid);
                tableId.PathId.ToProto(rec.MutablePathId());

                rec.SetSnapshotTxId(snapshot.TxId);
                rec.SetSnapshotStep(snapshot.Step);

                VectorIndexSettings settings;
                settings.set_vector_dimension(2);
                settings.set_vector_type(type);
                settings.set_metric(metric);
                *rec.MutableSettings() = settings;

                rec.SetSeed(seed);
                rec.SetM(m);
                rec.SetNBits(nbits);

                // rec.SetUpload(upload);

                rec.SetKMeansRounds(300);

                rec.SetParentFrom(parentFrom);
                rec.SetParentTo(parentTo);
                // rec.SetChild(parentTo + 1);

                rec.SetEmbeddingColumn("embedding");
                rec.AddDataColumns("data");

                rec.SetDatabaseName(kDatabaseName);

                // rec.SetOverlapClusters(overlapClusters);
                // rec.SetOverlapRatio(2);
                // rec.SetOverlapOutForeign(upload == NKikimrTxDataShard::EKMeansState::UPLOAD_MAIN_TO_BUILD ||
                //     upload == NKikimrTxDataShard::EKMeansState::UPLOAD_BUILD_TO_BUILD);

                rec.SetCodebookName(kCodebookTable);
                rec.SetOutputName(kPostingTable);

                rec.MutableScanSettings()->SetMaxBatchRows(maxBatchRows);
                rec.MutableSettings()->set_vector_dimension(4);
                rec.MutableSettings()->set_vector_type(type);
                rec.MutableSettings()->set_metric(metric);

                // if (keyRange) {
                //     keyRange->Serialize(*rec.MutableKeyRange());
                // }
            };
            fill(ev1);
            fill(ev2);

            runtime.SendToPipe(tid, sender, ev1.release(), 0, GetPipeConfigWithRetries());
            runtime.SendToPipe(tid, sender, ev2.release(), 0, GetPipeConfigWithRetries());

            TAutoPtr<IEventHandle> handle;
            auto reply = runtime.GrabEdgeEventRethrow<TEvDataShard::TEvLocalPqResponse>(handle);

            NYql::TIssues issues;
            NYql::IssuesFromMessage(reply->Record.GetIssues(), issues);
            UNIT_ASSERT_EQUAL_C(reply->Record.GetStatus(), NKikimrIndexBuilder::EBuildStatus::DONE,
                                issues.ToOneLineString());
            UNIT_ASSERT_EQUAL(reply->Record.GetIsEmpty(), expectEmpty);
        }

        auto codebook = ReadShardedTable(server, kCodebookTable);
        auto posting = ReadShardedTable(server, kPostingTable);
        Cerr << "Codebook:" << Endl;
        Cerr << codebook << Endl;
        Cerr << "Posting:" << Endl;
        Cerr << posting << Endl;
        return {std::move(codebook), std::move(posting)};
    }

    static void DropTable(Tests::TServer::TPtr server, TActorId sender, const TString& name)
    {
        ui64 txId = AsyncDropTable(server, sender, "/Root", name);
        WaitTxNotification(server, txId);
    }

    Y_UNIT_TEST(Default) {
        TPortManager pm;
        TServerSettings serverSettings(pm.GetPort(2134));
        serverSettings.SetDomainName("Root");

        Tests::TServer::TPtr server = new TServer(serverSettings);
        auto& runtime = *server->GetRuntime();
        auto sender = runtime.AllocateEdgeActor();

        runtime.SetLogPriority(NKikimrServices::TX_DATASHARD, NLog::PRI_DEBUG);
        runtime.SetLogPriority(NKikimrServices::BUILD_INDEX, NLog::PRI_TRACE);

        InitRoot(server, sender);

        TShardedTableOptions options;
        options.EnableOutOfOrder(true); // TODO(mbkkt) what is it?
        options.Shards(1);

        CreateBuildTable(server, sender, options, "table-build");
        // Upsert some initial values
        ExecSQL(server, sender,
            R"(
                UPSERT INTO `/Root/table-build` (`__ydb_parent`, `key`, `embedding`, `data`)
                VALUES
            )"
                    "(1, 1, \"\\x10\\x10\\x20\\x20\\x02\", \"one\"),"
                    "(1, 2, \"\\x30\\x30\\x40\\x40\\x02\", \"two\"),"
                    "(1, 3, \"\\x50\\x50\\x60\\x60\\x02\", \"three\"),"
                    "(1, 4, \"\\x70\\x70\\x80\\x80\\x02\", \"four\"),"
                    "(1, 5, \"\\x15\\x05\\x25\\x15\\x02\", \"five\"),"
                    "(1, 6, \"\\x25\\x25\\x45\\x35\\x02\", \"six\"),"
                    "(1, 7, \"\\x45\\x55\\x55\\x65\\x02\", \"seven\"),"
                    "(1, 8, \"\\x65\\x75\\x75\\x85\\x02\", \"eight\");"
        );

        auto create = [&] {
            CreateCodebookTable(server, sender, options);
            CreatePqPostingTable(server, sender, options);
        };
        create();
        auto recreate = [&] {
            DropTable(server, sender, "table-level");
            DropTable(server, sender, "table-posting");
            create();
        };

        ui64 m = 2, nbits = 2;
        ui64 seed;

        seed = 0;
        for (auto distance : {VectorIndexSettings::DISTANCE_MANHATTAN, VectorIndexSettings::DISTANCE_EUCLIDEAN}) {
            const auto [codebook, posting] = DoLocalPq(server, sender, 1, 1,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_MAIN_TO_BUILD,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, distance);

            TStringBuilder log;
            log << "codebook: " << codebook << Endl
                << "posting: " << posting << Endl;
            UNIT_FAIL(log);
            // UNIT_ASSERT_VALUES_EQUAL(codebook, "__ydb_parent = 1, __ydb_id = 1, __ydb_centroid = mm\2\n"
            //                                 "__ydb_parent = 1, __ydb_id = 2, __ydb_centroid = 11\2\n");
            // UNIT_ASSERT_VALUES_EQUAL(posting, "__ydb_parent = 1, key = 4, embedding = \x65\x65\2, data = four\n"
            //                                   "__ydb_parent = 1, key = 5, embedding = \x75\x75\2, data = five\n"
            //                                   "__ydb_parent = 2, key = 1, embedding = \x30\x30\2, data = one\n"
            //                                   "__ydb_parent = 2, key = 2, embedding = \x31\x31\2, data = two\n"
            //                                   "__ydb_parent = 2, key = 3, embedding = \x32\x32\2, data = three\n");
            recreate();
        }

        // seed = 111;
        // for (auto distance : {VectorIndexSettings::DISTANCE_MANHATTAN, VectorIndexSettings::DISTANCE_EUCLIDEAN}) {
        //     auto [level, posting] = DoLocalKMeans(server, sender, 0, 0, seed, k,
        //                                           NKikimrTxDataShard::EKMeansState::UPLOAD_MAIN_TO_BUILD,
        //                                           VectorIndexSettings::VECTOR_TYPE_UINT8, distance);
        //     UNIT_ASSERT_VALUES_EQUAL(level, "__ydb_parent = 0, __ydb_id = 1, __ydb_centroid = 11\2\n"
        //                                     "__ydb_parent = 0, __ydb_id = 2, __ydb_centroid = mm\2\n");
        //     UNIT_ASSERT_VALUES_EQUAL(posting, "__ydb_parent = 1, key = 1, embedding = \x30\x30\2, data = one\n"
        //                                       "__ydb_parent = 1, key = 2, embedding = \x31\x31\2, data = two\n"
        //                                       "__ydb_parent = 1, key = 3, embedding = \x32\x32\2, data = three\n"
        //                                       "__ydb_parent = 2, key = 4, embedding = \x65\x65\2, data = four\n"
        //                                       "__ydb_parent = 2, key = 5, embedding = \x75\x75\2, data = five\n");
        //     recreate();
        // }
        // seed = 32;
        // for (auto similarity : {VectorIndexSettings::SIMILARITY_INNER_PRODUCT, VectorIndexSettings::SIMILARITY_COSINE,
        //                         VectorIndexSettings::DISTANCE_COSINE})
        // {
        //     auto [level, posting] = DoLocalKMeans(server, sender, 0, 0, seed, k,
        //                                           NKikimrTxDataShard::EKMeansState::UPLOAD_MAIN_TO_BUILD,
        //                                           VectorIndexSettings::VECTOR_TYPE_UINT8, similarity);
        //     UNIT_ASSERT_VALUES_EQUAL(level, "__ydb_parent = 0, __ydb_id = 1, __ydb_centroid = II\2\n");
        //     UNIT_ASSERT_VALUES_EQUAL(posting, "__ydb_parent = 1, key = 1, embedding = \x30\x30\2, data = one\n"
        //                                       "__ydb_parent = 1, key = 2, embedding = \x31\x31\2, data = two\n"
        //                                       "__ydb_parent = 1, key = 3, embedding = \x32\x32\2, data = three\n"
        //                                       "__ydb_parent = 1, key = 4, embedding = \x65\x65\2, data = four\n"
        //                                       "__ydb_parent = 1, key = 5, embedding = \x75\x75\2, data = five\n");
        //     recreate();
        // }
    }
}

}
