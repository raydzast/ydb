#include "ut_helpers.h"

#include <library/cpp/testing/unittest/registar.h>
#include <ydb/core/testlib/test_client.h>

#include <atomic>

namespace NKikimr {
using namespace Tests;
using Ydb::Table::VectorIndexSettings;

static std::atomic<ui64> sId = 1;
static const TString kDatabaseName = "/Root";
static const TString kMainTable = "/Root/table-main";
static const TString kBuildTable = "/Root/table-build";
static const TString kCodebookTable = "/Root/table-codebook";
static const TString kPostingTable = "/Root/table-posting";

Y_UNIT_TEST_SUITE(TTxDataShardLocalPqScan) {

    static std::tuple<TString, TString> DoLocalPq(
        Tests::TServer::TPtr server, TActorId sender,
        NTableIndex::NKMeans::TClusterId parentFrom,
        NTableIndex::NKMeans::TClusterId parentTo,
        NKikimrTxDataShard::EKMeansState /*uploadState*/,
        ui64 seed, ui64 m, ui64 nbits,
        VectorIndexSettings::VectorType type, VectorIndexSettings::Metric metric,
        ui32 maxBatchRows = 50000/*, ui32 overlapClusters = 0*/, bool expectEmpty = false/*,
        std::optional<TSerializedTableRange> keyRange = {}*/
    ) {
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

                // TODO(raydzast): shall I add upload state?
                // rec.SetUpload(uploadState);

                rec.SetKMeansRounds(300);

                rec.SetParentFrom(parentFrom);
                rec.SetParentTo(parentTo);

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
        Cerr << codebook.Quote() << Endl;
        Cerr << "Posting:" << Endl;
        Cerr << posting.Quote() << Endl;
        return {std::move(codebook), std::move(posting)};
    }

    static void DropTable(Tests::TServer::TPtr server, TActorId sender, const TString& name) {
        ui64 txId = AsyncDropTable(server, sender, "/Root", name);
        WaitTxNotification(server, txId);
    }

    Y_UNIT_TEST(Empty) {
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

        auto create = [&] {
            CreateCodebookTable(server, sender, options);
            CreatePqPostingTable(server, sender, options);
        };
        create();
        auto recreate = [&] {
            DropTable(server, sender, "table-codebook");
            DropTable(server, sender, "table-posting");
            create();
        };

        const ui64 m = 2, nbits = 2;
        for (const ui64 seed : {0, 111, 32}) {
            const auto [codebook, posting] = DoLocalPq(server, sender, 0, 0,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_BUILD_TO_POSTING,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, VectorIndexSettings::DISTANCE_EUCLIDEAN,
                                                  50000, true);

            UNIT_ASSERT_VALUES_EQUAL(codebook, ""_sb);
            UNIT_ASSERT_VALUES_EQUAL(posting, ""_sb);

            recreate();
        }
    }

    Y_UNIT_TEST(MainToPosting) {
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

        CreateBuildTable(server, sender, options, "table-main");
        // Upsert some initial values
        ExecSQL(server, sender,
            R"(
                UPSERT INTO `/Root/table-main` (`__ydb_parent`, `key`, `embedding`, `data`)
                VALUES
            )"
                    "(1, \"\\x10\\x10\\x20\\x20\\x02\", \"one\"),"
                    "(2, \"\\x30\\x30\\x40\\x40\\x02\", \"two\"),"
                    "(3, \"\\x50\\x50\\x60\\x60\\x02\", \"three\"),"
                    "(4, \"\\x70\\x70\\x80\\x80\\x02\", \"four\"),"
                    "(5, \"\\x15\\x05\\x55\\x65\\x02\", \"five\"),"
                    "(6, \"\\x25\\x25\\x45\\x35\\x02\", \"six\"),"
                    "(7, \"\\x45\\x55\\x75\\x85\\x02\", \"seven\"),"
                    "(8, \"\\x65\\x75\\x25\\x15\\x02\", \"eight\");"
        );

        auto create = [&] {
            CreateCodebookTable(server, sender, options);
            CreatePqPostingTable(server, sender, options);
        };
        create();
        auto recreate = [&] {
            DropTable(server, sender, "table-codebook");
            DropTable(server, sender, "table-posting");
            create();
        };

        const ui64 m = 2, nbits = 2;
        {
            const ui64 seed = 0;
            const auto [codebook, posting] = DoLocalPq(server, sender, 0, 0,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_MAIN_TO_POSTING,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, VectorIndexSettings::DISTANCE_EUCLIDEAN);

            UNIT_ASSERT_VALUES_EQUAL(
                codebook,
                "__ydb_segment = \0, __ydb_code = \0, __ydb_centroid = jr\2\n"
                "__ydb_segment = \0, __ydb_code = \1, __ydb_centroid = 00\2\n"
                "__ydb_segment = \0, __ydb_code = \2, __ydb_centroid = JR\2\n"
                "__ydb_segment = \0, __ydb_code = \3, __ydb_centroid = \x18\x13\2\n"
                "__ydb_segment = \1, __ydb_code = \0, __ydb_centroid = \"\x1A\2\n"
                "__ydb_segment = \1, __ydb_code = \1, __ydb_centroid = @@\2\n"
                "__ydb_segment = \1, __ydb_code = \2, __ydb_centroid = jr\2\n"
                "__ydb_segment = \1, __ydb_code = \3, __ydb_centroid = E5\2\n"_sb
            );
            UNIT_ASSERT_VALUES_EQUAL(
                posting,
                "key = 1, __ydb_codes = \3\0\0\x82, data = one\n"
                "key = 2, __ydb_codes = \1\1\0\x82, data = two\n"
                "key = 3, __ydb_codes = \2\2\0\x82, data = three\n"
                "key = 4, __ydb_codes = \0\2\0\x82, data = four\n"
                "key = 5, __ydb_codes = \3\2\0\x82, data = five\n"
                "key = 6, __ydb_codes = \1\3\0\x82, data = six\n"
                "key = 7, __ydb_codes = \2\2\0\x82, data = seven\n"
                "key = 8, __ydb_codes = \0\0\0\x82, data = eight\n"_sb
            );

            recreate();
        }
        {
            const ui64 seed = 111;
            const auto [codebook, posting] = DoLocalPq(server, sender, 0, 0,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_MAIN_TO_POSTING,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, VectorIndexSettings::DISTANCE_EUCLIDEAN);

            UNIT_ASSERT_VALUES_EQUAL(
                codebook,
                "__ydb_segment = \0, __ydb_code = \0, __ydb_centroid = \x12\n\2\n"
                "__ydb_segment = \0, __ydb_code = \1, __ydb_centroid = 00\2\n"
                "__ydb_segment = \0, __ydb_code = \2, __ydb_centroid = Zb\2\n"
                "__ydb_segment = \0, __ydb_code = \3, __ydb_centroid = %%\2\n"
                "__ydb_segment = \1, __ydb_code = \0, __ydb_centroid = \"\x1A\2\n"
                "__ydb_segment = \1, __ydb_code = \1, __ydb_centroid = @@\2\n"
                "__ydb_segment = \1, __ydb_code = \2, __ydb_centroid = jr\2\n"
                "__ydb_segment = \1, __ydb_code = \3, __ydb_centroid = E5\2\n"_sb
            );
            UNIT_ASSERT_VALUES_EQUAL(
                posting,
                "key = 1, __ydb_codes = \0\0\0\x82, data = one\n"
                "key = 2, __ydb_codes = \1\1\0\x82, data = two\n"
                "key = 3, __ydb_codes = \2\2\0\x82, data = three\n"
                "key = 4, __ydb_codes = \2\2\0\x82, data = four\n"
                "key = 5, __ydb_codes = \0\2\0\x82, data = five\n"
                "key = 6, __ydb_codes = \3\3\0\x82, data = six\n"
                "key = 7, __ydb_codes = \2\2\0\x82, data = seven\n"
                "key = 8, __ydb_codes = \2\0\0\x82, data = eight\n"_sb
            );

            recreate();
        }
        {
            const ui64 seed = 32;
            const auto [codebook, posting] = DoLocalPq(server, sender, 0, 0,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_MAIN_TO_POSTING,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, VectorIndexSettings::DISTANCE_EUCLIDEAN);

            UNIT_ASSERT_VALUES_EQUAL(
                codebook,
                "__ydb_segment = \0, __ydb_code = \0, __ydb_centroid = \x12\n\2\n"
                "__ydb_segment = \0, __ydb_code = \1, __ydb_centroid = **\2\n"
                "__ydb_segment = \0, __ydb_code = \2, __ydb_centroid = JR\2\n"
                "__ydb_segment = \0, __ydb_code = \3, __ydb_centroid = jr\2\n"
                "__ydb_segment = \1, __ydb_code = \0, __ydb_centroid = Ue\2\n"
                "__ydb_segment = \1, __ydb_code = \1, __ydb_centroid = B:\2\n"
                "__ydb_segment = \1, __ydb_code = \2, __ydb_centroid = qw\2\n"
                "__ydb_segment = \1, __ydb_code = \3, __ydb_centroid = \"\x1A\2\n"_sb
            );
            UNIT_ASSERT_VALUES_EQUAL(
                posting,
                "key = 1, __ydb_codes = \0\3\0\x82, data = one\n"
                "key = 2, __ydb_codes = \1\1\0\x82, data = two\n"
                "key = 3, __ydb_codes = \2\0\0\x82, data = three\n"
                "key = 4, __ydb_codes = \3\2\0\x82, data = four\n"
                "key = 5, __ydb_codes = \0\0\0\x82, data = five\n"
                "key = 6, __ydb_codes = \1\1\0\x82, data = six\n"
                "key = 7, __ydb_codes = \2\2\0\x82, data = seven\n"
                "key = 8, __ydb_codes = \3\3\0\x82, data = eight\n"_sb
            );

            recreate();
        }
    }

    Y_UNIT_TEST(BuildToPosting) {
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
                    "(1, 5, \"\\x15\\x05\\x55\\x65\\x02\", \"five\"),"
                    "(1, 6, \"\\x25\\x25\\x45\\x35\\x02\", \"six\"),"
                    "(1, 7, \"\\x45\\x55\\x75\\x85\\x02\", \"seven\"),"
                    "(1, 8, \"\\x65\\x75\\x25\\x15\\x02\", \"eight\");"
        );

        auto create = [&] {
            CreateCodebookTable(server, sender, options);
            CreatePqPostingTable(server, sender, options);
        };
        create();
        auto recreate = [&] {
            DropTable(server, sender, "table-codebook");
            DropTable(server, sender, "table-posting");
            create();
        };

        const ui64 m = 2, nbits = 2;
        {
            const ui64 seed = 0;
            const auto [codebook, posting] = DoLocalPq(server, sender, 1, 1,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_BUILD_TO_POSTING,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, VectorIndexSettings::DISTANCE_EUCLIDEAN);

            UNIT_ASSERT_VALUES_EQUAL(
                codebook,
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \0, __ydb_centroid = jr\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \1, __ydb_centroid = **\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \2, __ydb_centroid = JR\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \3, __ydb_centroid = \x12\n\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \0, __ydb_centroid = \"\x1A\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \1, __ydb_centroid = @@\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \2, __ydb_centroid = jr\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \3, __ydb_centroid = E5\2\n"_sb
            );
            UNIT_ASSERT_VALUES_EQUAL(
                posting,
                "__ydb_parent = 1, key = 1, __ydb_codes = \3\0\0\x82, data = one\n"
                "__ydb_parent = 1, key = 2, __ydb_codes = \1\1\0\x82, data = two\n"
                "__ydb_parent = 1, key = 3, __ydb_codes = \2\2\0\x82, data = three\n"
                "__ydb_parent = 1, key = 4, __ydb_codes = \0\2\0\x82, data = four\n"
                "__ydb_parent = 1, key = 5, __ydb_codes = \3\2\0\x82, data = five\n"
                "__ydb_parent = 1, key = 6, __ydb_codes = \1\3\0\x82, data = six\n"
                "__ydb_parent = 1, key = 7, __ydb_codes = \2\2\0\x82, data = seven\n"
                "__ydb_parent = 1, key = 8, __ydb_codes = \0\0\0\x82, data = eight\n"_sb
            );

            recreate();
        }
        {
            const ui64 seed = 111;
            const auto [codebook, posting] = DoLocalPq(server, sender, 1, 1,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_BUILD_TO_POSTING,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, VectorIndexSettings::DISTANCE_EUCLIDEAN);

            UNIT_ASSERT_VALUES_EQUAL(
                codebook,
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \0, __ydb_centroid = \x12\n\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \1, __ydb_centroid = 00\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \2, __ydb_centroid = Zb\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \3, __ydb_centroid = %%\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \0, __ydb_centroid = \"\x1A\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \1, __ydb_centroid = @@\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \2, __ydb_centroid = jr\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \3, __ydb_centroid = E5\2\n"_sb
            );
            UNIT_ASSERT_VALUES_EQUAL(
                posting,
                "__ydb_parent = 1, key = 1, __ydb_codes = \0\0\0\x82, data = one\n"
                "__ydb_parent = 1, key = 2, __ydb_codes = \1\1\0\x82, data = two\n"
                "__ydb_parent = 1, key = 3, __ydb_codes = \2\2\0\x82, data = three\n"
                "__ydb_parent = 1, key = 4, __ydb_codes = \2\2\0\x82, data = four\n"
                "__ydb_parent = 1, key = 5, __ydb_codes = \0\2\0\x82, data = five\n"
                "__ydb_parent = 1, key = 6, __ydb_codes = \3\3\0\x82, data = six\n"
                "__ydb_parent = 1, key = 7, __ydb_codes = \2\2\0\x82, data = seven\n"
                "__ydb_parent = 1, key = 8, __ydb_codes = \2\0\0\x82, data = eight\n"_sb
            );

            recreate();
        }
        {
            const ui64 seed = 32;
            const auto [codebook, posting] = DoLocalPq(server, sender, 1, 1,
                                                  NKikimrTxDataShard::EKMeansState::UPLOAD_BUILD_TO_POSTING,
                                                  seed, m, nbits,
                                                  VectorIndexSettings::VECTOR_TYPE_UINT8, VectorIndexSettings::DISTANCE_EUCLIDEAN);

            UNIT_ASSERT_VALUES_EQUAL(
                codebook,
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \0, __ydb_centroid = \x12\n\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \1, __ydb_centroid = **\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \2, __ydb_centroid = JR\2\n"
                "__ydb_parent = 1, __ydb_segment = \0, __ydb_code = \3, __ydb_centroid = jr\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \0, __ydb_centroid = Zb\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \1, __ydb_centroid = B:\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \2, __ydb_centroid = z\x82\2\n"
                "__ydb_parent = 1, __ydb_segment = \1, __ydb_code = \3, __ydb_centroid = \"\x1A\2\n"_sb
            );
            UNIT_ASSERT_VALUES_EQUAL(
                posting,
                "__ydb_parent = 1, key = 1, __ydb_codes = \0\3\0\x82, data = one\n"
                "__ydb_parent = 1, key = 2, __ydb_codes = \1\1\0\x82, data = two\n"
                "__ydb_parent = 1, key = 3, __ydb_codes = \2\0\0\x82, data = three\n"
                "__ydb_parent = 1, key = 4, __ydb_codes = \3\2\0\x82, data = four\n"
                "__ydb_parent = 1, key = 5, __ydb_codes = \0\0\0\x82, data = five\n"
                "__ydb_parent = 1, key = 6, __ydb_codes = \1\1\0\x82, data = six\n"
                "__ydb_parent = 1, key = 7, __ydb_codes = \2\2\0\x82, data = seven\n"
                "__ydb_parent = 1, key = 8, __ydb_codes = \3\3\0\x82, data = eight\n"_sb
            );

            recreate();
        }
    }

    Y_UNIT_TEST(MainToPostingWithForeign) {
        // TODO(raydzast)
        UNIT_FAIL("Not implemented");
    }

    Y_UNIT_TEST(BuildToPostingWithForeign) {
        // TODO(raydzast)
        UNIT_FAIL("Not implemented");
    }

    Y_UNIT_TEST(MainToPostingWithKeyRange) {
        // TODO(raydzast)
        UNIT_FAIL("Not implemented");
    }

    Y_UNIT_TEST(BuildToPostingWithKeyRange) {
        // TODO(raydzast)
        UNIT_FAIL("Not implemented");
    }

    Y_UNIT_TEST(VariousNBits) {
        // TODO(raydzast)
        UNIT_FAIL("Not implemented");
    }

    Y_UNIT_TEST(FewLines) {
        // TODO(raydzast)
        UNIT_FAIL("Not implemented");
    }

    Y_UNIT_TEST(OneSubspaceKMeansConvergedFasterThanOther) {
        // TODO(raydzast)
        UNIT_FAIL("Not implemented");
    }

}

}
