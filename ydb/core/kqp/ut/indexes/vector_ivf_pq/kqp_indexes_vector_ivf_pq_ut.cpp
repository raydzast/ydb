#include <library/cpp/testing/unittest/registar.h>
#include <ydb/core/kqp/ut/common/kqp_ut_common.h>

namespace NKikimr {
    namespace NKqp {

        using namespace NYdb;

        Y_UNIT_TEST_SUITE(KqpVectorIndexesIvfPq) {

            TKikimrRunner Kikimr() {
                TTestLogSettings logSettings = TTestLogSettings()
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::FLAT_TX_SCHEMESHARD, NActors::NLog::EPriority::PRI_DEBUG)
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::TX_DATASHARD, NActors::NLog::EPriority::PRI_DEBUG)
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::TABLET_EXECUTOR, NActors::NLog::EPriority::PRI_DEBUG)
                                                   .AddLogPriority(NKikimrServices::EServiceKikimr::BUILD_INDEX, NActors::NLog::EPriority::PRI_DEBUG);

                NKikimrConfig::TFeatureFlags featureFlags;
                featureFlags.SetEnableIvfPqIndex(true);

                auto settings = TKikimrSettings()
                                    .SetFeatureFlags(featureFlags)
                                    .SetVerbose(true)
                                    .SetLogSettings(std::move(logSettings));
                settings.AppConfig.MutableTableServiceConfig()->SetBackportMode(NKikimrConfig::TTableServiceConfig_EBackportMode_All);
                return TKikimrRunner(settings);
            }

            void CreateMain(NQuery::TQueryClient& db) {
                const TString query = R"sql(
                    CREATE TABLE `/Root/main` (
                        `Key` Uint64,
                        `Embedding` String,
                        `Data` String,
                        PRIMARY KEY (Key)
                    );
                )sql";
                const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
            }

            TResultSet ReadIndex(NQuery::TQueryClient& db, const char* table = "indexImplTable", const char* columns = "*") {
                TString query = Sprintf(R"sql(
                    SELECT %s FROM `/Root/main/ivf_pq_index/%s`;
                )sql", columns, table);
                auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                return result.GetResultSet(0);
            }

            Y_UNIT_TEST(AddIndexIvfPq) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);

                {
                    const TString query = R"(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        VALUES
                    )"
                                          "(1, \"\\x10\\x10\\x20\\x20\\x02\", \"one\"),"
                                          "(2, \"\\x30\\x30\\x40\\x40\\x02\", \"two\"),"
                                          "(3, \"\\x50\\x50\\x60\\x60\\x02\", \"three\"),"
                                          "(4, \"\\x70\\x70\\x80\\x80\\x02\", \"four\"),"
                                          "(5, \"\\x15\\x05\\x55\\x65\\x02\", \"five\"),"
                                          "(6, \"\\x25\\x25\\x45\\x35\\x02\", \"six\"),"
                                          "(7, \"\\x45\\x55\\x75\\x85\\x02\", \"seven\"),"
                                          "(8, \"\\x65\\x75\\x25\\x15\\x02\", \"eight\"),"
                                          "(9, \"\\x20\\x20\\x10\\x10\\x02\", \"nine\"),"
                                          "(10, \"\\x40\\x40\\x30\\x30\\x02\", \"ten\"),"
                                          "(11, \"\\x72\\x72\\x78\\x78\\x02\", \"eleven\");";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                {
                    const TString query = R"sql(
                        ALTER TABLE `/Root/main`
                            ADD INDEX `ivf_pq_index`
                                GLOBAL SYNC
                                USING vector_ivf_pq
                                ON (`Embedding`)
                                WITH (
                                    vector_dimension=4,
                                    vector_type=uint8,
                                    distance=euclidean,
                                    kmeans_tree_clusters=2,
                                    kmeans_tree_levels=1,
                                    pq_m=2,
                                    pq_nbits=2
                                );
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                const auto level = ReadIndex(db, "indexImplLevelTable", "`__ydb_parent`, `__ydb_id`, `__ydb_centroid`");
                CompareYson(R"([
                    [0u;1u;"]asw\2";];
                    [0u;2u;"--2/\2";];
                ])", NYdb::FormatResultSetYson(level));

                const auto codebook = ReadIndex(db, "indexImplCodebookTable", "`__ydb_parent`, `__ydb_segment`, `__ydb_code`, `__ydb_centroid`");
                CompareYson(R"([
                    [1u;0u;0u;"PP\2"];
                    [1u;0u;1u;"pp\2"];
                    [1u;0u;2u;"EU\2"];
                    [1u;0u;3u;"rr\2"];
                    [1u;1u;0u;"``\2"];
                    [1u;1u;1u;"\x80\x80\2"];
                    [1u;1u;2u;"u\x85\2"];
                    [1u;1u;3u;"xx\2"];
                    [2u;0u;0u;"eu\2"];
                    [2u;0u;1u;"''\2"];
                    [2u;0u;2u;"\x12\n\2"];
                    [2u;0u;3u;"@@\2"];
                    [2u;1u;0u;"\x1C\x17\2"];
                    [2u;1u;1u;"B:\2"];
                    [2u;1u;2u;"Ue\2"];
                    [2u;1u;3u;"00\2"];
                ])", NYdb::FormatResultSetYson(codebook));
                UNIT_ASSERT_VALUES_EQUAL(codebook.RowsCount(), 16);

                const auto posting = ReadIndex(db, "indexImplPostingTable", "`__ydb_parent`, `Key`, `__ydb_codes`");
                CompareYson(R"([
                    [1u;[3u];["\0\x04\x82"]];
                    [1u;[4u];["\x05\x04\x82"]];
                    [1u;[7u];["\x0A\x04\x82"]];
                    [1u;[11u];["\x0F\x04\x82"]];
                    [2u;[1u];["\x02\x04\x82"]];
                    [2u;[2u];["\x05\x04\x82"]];
                    [2u;[5u];["\x0A\x04\x82"]];
                    [2u;[6u];["\x05\x04\x82"]];
                    [2u;[8u];["\0\x04\x82"]];
                    [2u;[9u];["\x01\x04\x82"]];
                    [2u;[10u];["\x0F\x04\x82"]];
                ])", NYdb::FormatResultSetYson(posting));
            }

        } // Y_UNIT_TEST_SUITE(KqpVectorIndexesIvfPq)

    } // namespace NKqp
} // namespace NKikimr
