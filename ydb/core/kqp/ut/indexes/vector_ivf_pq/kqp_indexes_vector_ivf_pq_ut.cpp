#include <library/cpp/testing/unittest/registar.h>
#include <ydb/core/kqp/ut/common/kqp_ut_common.h>

#include <ydb/public/sdk/cpp/include/ydb-cpp-sdk/client/table/table.h>

namespace NKikimr {
    namespace NKqp {

        using namespace NYdb;
        using namespace NYdb::NTable;

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
                    // IVF_PQ supports float vectors only; use Untag(ToBinaryStringFloat(...)).
                    const TString query = R"(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        VALUES
                            (1u, Untag(Knn::ToBinaryStringFloat([1.0f, 1.0f, 2.0f, 2.0f]), "FloatVector"), "one"),
                            (2u, Untag(Knn::ToBinaryStringFloat([3.0f, 3.0f, 4.0f, 4.0f]), "FloatVector"), "two"),
                            (3u, Untag(Knn::ToBinaryStringFloat([5.0f, 5.0f, 6.0f, 6.0f]), "FloatVector"), "three"),
                            (4u, Untag(Knn::ToBinaryStringFloat([7.0f, 7.0f, 8.0f, 8.0f]), "FloatVector"), "four"),
                            (5u, Untag(Knn::ToBinaryStringFloat([1.5f, 0.5f, 5.5f, 6.5f]), "FloatVector"), "five"),
                            (6u, Untag(Knn::ToBinaryStringFloat([2.5f, 2.5f, 4.5f, 3.5f]), "FloatVector"), "six"),
                            (7u, Untag(Knn::ToBinaryStringFloat([4.5f, 5.5f, 7.5f, 8.5f]), "FloatVector"), "seven"),
                            (8u, Untag(Knn::ToBinaryStringFloat([6.5f, 7.5f, 2.5f, 1.5f]), "FloatVector"), "eight"),
                            (9u, Untag(Knn::ToBinaryStringFloat([2.0f, 2.0f, 1.0f, 1.0f]), "FloatVector"), "nine"),
                            (10u, Untag(Knn::ToBinaryStringFloat([4.0f, 4.0f, 3.0f, 3.0f]), "FloatVector"), "ten"),
                            (11u, Untag(Knn::ToBinaryStringFloat([7.2f, 7.2f, 7.8f, 7.8f]), "FloatVector"), "eleven");
                    )";
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
                                    vector_type=float,
                                    distance=euclidean,
                                    kmeans_tree_clusters=2,
                                    kmeans_tree_levels=1,
                                    subspaces=2,
                                    subspace_bits=2
                                );
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                const auto level = ReadIndex(db, "indexImplLevelTable", "`__ydb_parent`, `__ydb_id`");
                UNIT_ASSERT_VALUES_EQUAL(level.RowsCount(), 2u);
                THashSet<ui64> levelIds;
                {
                    TResultSetParser parser(level);
                    while (parser.TryNextRow()) {
                        UNIT_ASSERT_VALUES_EQUAL(parser.ColumnParser(0).GetUint64(), 0u);
                        levelIds.insert(parser.ColumnParser(1).GetUint64());
                    }
                }
                UNIT_ASSERT(levelIds.contains(1u));
                UNIT_ASSERT(levelIds.contains(2u));

                const auto codebook = ReadIndex(db, "indexImplCodebookTable", "`__ydb_parent`");
                UNIT_ASSERT_VALUES_EQUAL(codebook.RowsCount(), 8u);

                const auto posting = ReadIndex(db, "indexImplPostingTable", "`__ydb_parent`, `Key`");
                UNIT_ASSERT_VALUES_EQUAL(posting.RowsCount(), 11u);
            }

            Y_UNIT_TEST(AnnQueryE2e) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetQueryClient();

                CreateMain(db);

                {
                    const TString query = R"(
                        UPSERT INTO `/Root/main` (`Key`, `Embedding`, `Data`)
                        VALUES
                            (1u, Untag(Knn::ToBinaryStringFloat([1.0f, 1.0f, 2.0f, 2.0f]), "FloatVector"), "one"),
                            (2u, Untag(Knn::ToBinaryStringFloat([3.0f, 3.0f, 4.0f, 4.0f]), "FloatVector"), "two"),
                            (3u, Untag(Knn::ToBinaryStringFloat([5.0f, 5.0f, 6.0f, 6.0f]), "FloatVector"), "three"),
                            (4u, Untag(Knn::ToBinaryStringFloat([7.0f, 7.0f, 8.0f, 8.0f]), "FloatVector"), "four"),
                            (5u, Untag(Knn::ToBinaryStringFloat([1.5f, 0.5f, 5.5f, 6.5f]), "FloatVector"), "five"),
                            (6u, Untag(Knn::ToBinaryStringFloat([2.5f, 2.5f, 4.5f, 3.5f]), "FloatVector"), "six"),
                            (7u, Untag(Knn::ToBinaryStringFloat([4.5f, 5.5f, 7.5f, 8.5f]), "FloatVector"), "seven"),
                            (8u, Untag(Knn::ToBinaryStringFloat([6.5f, 7.5f, 2.5f, 1.5f]), "FloatVector"), "eight"),
                            (9u, Untag(Knn::ToBinaryStringFloat([2.0f, 2.0f, 1.0f, 1.0f]), "FloatVector"), "nine"),
                            (10u, Untag(Knn::ToBinaryStringFloat([4.0f, 4.0f, 3.0f, 3.0f]), "FloatVector"), "ten"),
                            (11u, Untag(Knn::ToBinaryStringFloat([7.2f, 7.2f, 7.8f, 7.8f]), "FloatVector"), "eleven");
                    )";
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
                                    vector_type=float,
                                    distance=euclidean,
                                    kmeans_tree_clusters=2,
                                    kmeans_tree_levels=1,
                                    subspaces=2,
                                    subspace_bits=2
                                );
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                THashSet<ui64> bruteForceKeys;
                {
                    const TString query = R"sql(
                        $target = Untag(Knn::ToBinaryStringFloat([0.0f, 0.0f, 0.0f, 0.0f]), "FloatVector");
                        SELECT `Key` FROM `/Root/main`
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )sql";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    TResultSetParser parser(result.GetResultSet(0));
                    while (parser.TryNextRow()) {
                        const auto key = parser.ColumnParser(0).GetOptionalUint64();
                        UNIT_ASSERT_C(key.has_value(), "Key must be present");
                        bruteForceKeys.insert(*key);
                    }
                }

                THashSet<ui64> annKeys;
                {
                    const TString query = R"(
                        PRAGMA ydb.KMeansTreeSearchTopSize = "2";
                        $target = Untag(Knn::ToBinaryStringFloat([0.0f, 0.0f, 0.0f, 0.0f]), "FloatVector");
                        SELECT `Key` FROM `/Root/main`
                        VIEW ivf_pq_index
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )";
                    const auto result = db.ExecuteQuery(query, NYdb::NQuery::TTxControl::NoTx()).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                    TResultSetParser parser(result.GetResultSet(0));
                    while (parser.TryNextRow()) {
                        const auto key = parser.ColumnParser(0).GetOptionalUint64();
                        UNIT_ASSERT_C(key.has_value(), "Key must be present");
                        annKeys.insert(*key);
                    }
                }

                ui32 overlap = 0;
                for (const auto key : annKeys) {
                    if (bruteForceKeys.contains(key)) {
                        ++overlap;
                    }
                }
                UNIT_ASSERT_C(overlap >= 1, "ANN recall@3 should overlap brute-force top-3 on the fixture");
            }

            // With EnableIvfPqIndex on, logical rewrite must produce KqpBuildPqDistanceTable in the AST.
            Y_UNIT_TEST(AnnQueryRewritePlan) {
                auto kikimr = Kikimr();
                auto db = kikimr.GetTableClient();
                auto session = db.CreateSession().GetValueSync().GetSession();

                {
                    const TString query = R"sql(
                        CREATE TABLE `/Root/main` (
                            `Key` Uint64,
                            `Embedding` String,
                            `Data` String,
                            PRIMARY KEY (Key)
                        );
                    )sql";
                    const auto result = session.ExecuteSchemeQuery(query).ExtractValueSync();
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
                                    vector_type=float,
                                    distance=euclidean,
                                    kmeans_tree_clusters=2,
                                    kmeans_tree_levels=1,
                                    subspaces=2,
                                    subspace_bits=2
                                );
                    )sql";
                    const auto result = session.ExecuteSchemeQuery(query).ExtractValueSync();
                    UNIT_ASSERT_VALUES_EQUAL_C(result.GetStatus(), EStatus::SUCCESS, result.GetIssues().ToString());
                }

                {
                    // 4 float zeros + FloatVector format tag (0x01) at the end.
                    const TString query = R"(
                        $target = "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01";
                        SELECT `Key` FROM `/Root/main`
                        VIEW ivf_pq_index
                        ORDER BY Knn::EuclideanDistance(`Embedding`, $target) ASC
                        LIMIT 3;
                    )";
                    const auto result = session.ExplainDataQuery(query).ExtractValueSync();
                    UNIT_ASSERT_C(result.IsSuccess(), result.GetIssues().ToString());
                    const auto& ast = result.GetAst();
                    UNIT_ASSERT_C(ast.find("KqpBuildPqDistanceTable") != std::string::npos, ast);
                    UNIT_ASSERT_C(ast.find("indexImplCodebookTable") != std::string::npos, ast);
                    UNIT_ASSERT_C(ast.find("IvfPqDistanceTables") != std::string::npos, ast);
                }
            }

        } // Y_UNIT_TEST_SUITE(KqpVectorIndexesIvfPq)

    } // namespace NKqp
} // namespace NKikimr
