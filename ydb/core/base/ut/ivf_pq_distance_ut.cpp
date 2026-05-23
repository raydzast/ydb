#include "ivf_pq.h"

#include <ydb/library/yql/udfs/common/knn/knn-serializer-shared.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/random/fast.h>
#include <util/stream/str.h>

#include <cmath>
#include <limits>

namespace NKikimr::NIvfPq {

namespace {

    TString SerializeFloatVector(const TVector<float>& values) {
        TString result;
        TStringOutput output(result);
        NKnnVectorSerialization::TSerializer<float> serializer(&output);
        for (const float v : values) {
            serializer.HandleElement(v);
        }
        serializer.Finish();
        return result;
    }

    TVector<float> DeserializeFloatVector(const TStringBuf data) {
        TVector<float> result;
        NKnnVectorSerialization::TDeserializer<float> deserializer(data);
        result.reserve(deserializer.GetElementCount());
        deserializer.DoDeserialize([&](const float& v) {
            result.push_back(v);
        });
        return result;
    }

    float SquaredEuclidean(const float* a, const float* b, size_t len) {
        float result = 0;
        for (size_t i = 0; i < len; ++i) {
            const float diff = a[i] - b[i];
            result += diff * diff;
        }
        return result;
    }

    void AssertNear(float expected, float actual, float eps = 1e-4f, const TString& message = {}) {
        UNIT_ASSERT_C(std::fabs(expected - actual) <= eps,
            "expected " << expected << " got " << actual << " (eps=" << eps << ") " << message);
    }

} // namespace

Y_UNIT_TEST_SUITE(NIvfPqDistance) {

    Y_UNIT_TEST(BuildPqDistanceTable_TrivialCase) {
        // M=2 sub-spaces, each of dim 2; nbits=1 -> codeCount=2.
        // residual = [1.0, 2.0,  3.0, 4.0]
        // codebook[0] = [(0,0), (1,2)]
        // codebook[1] = [(0,0), (3,4)]
        // expected:
        //   sub_residual[0] = (1,2)  -> dist to (0,0) = 5,    dist to (1,2) = 0
        //   sub_residual[1] = (3,4)  -> dist to (0,0) = 25,   dist to (3,4) = 0
        const TString residual = SerializeFloatVector({1.f, 2.f, 3.f, 4.f});
        const TString codebook00 = SerializeFloatVector({0.f, 0.f});
        const TString codebook01 = SerializeFloatVector({1.f, 2.f});
        const TString codebook10 = SerializeFloatVector({0.f, 0.f});
        const TString codebook11 = SerializeFloatVector({3.f, 4.f});

        const TVector<TVector<TStringBuf>> codebook = {
            {codebook00, codebook01},
            {codebook10, codebook11},
        };

        const TString distanceTableBlob = BuildPqDistanceTable(residual, codebook, /*m=*/2, /*nbits=*/1);

        const TVector<float> distanceTable = DeserializeFloatVector(distanceTableBlob);
        UNIT_ASSERT_VALUES_EQUAL(distanceTable.size(), 4u);
        AssertNear(5.f,  distanceTable[0], 1e-4f, "distanceTable[0,0]");
        AssertNear(0.f,  distanceTable[1], 1e-4f, "distanceTable[0,1]");
        AssertNear(25.f, distanceTable[2], 1e-4f, "distanceTable[1,0]");
        AssertNear(0.f,  distanceTable[3], 1e-4f, "distanceTable[1,1]");
    }

    Y_UNIT_TEST(BuildPqDistanceTable_MatchesBruteforce) {
        // Random residual and codebook; distanceTable entries must match a manual
        // squared L2 between the corresponding sub-vectors.
        constexpr ui32 m = 4;
        constexpr ui32 nbits = 4;
        constexpr ui32 codeCount = 1u << nbits;
        constexpr ui32 subDim = 5;
        constexpr ui32 totalDim = m * subDim;

        TReallyFastRng32 rng(42);
        TVector<float> residualValues(totalDim);
        for (auto& v : residualValues) {
            v = rng.GenRandReal2() * 2.f - 1.f;
        }
        const TString residual = SerializeFloatVector(residualValues);

        TVector<TVector<TString>> codebookStorage(m);
        TVector<TVector<TStringBuf>> codebook(m);
        for (ui32 mIdx = 0; mIdx < m; ++mIdx) {
            codebookStorage[mIdx].reserve(codeCount);
            codebook[mIdx].reserve(codeCount);
            for (ui32 codeIdx = 0; codeIdx < codeCount; ++codeIdx) {
                TVector<float> sub(subDim);
                for (auto& v : sub) {
                    v = rng.GenRandReal2() * 2.f - 1.f;
                }
                codebookStorage[mIdx].push_back(SerializeFloatVector(sub));
                codebook[mIdx].push_back(codebookStorage[mIdx].back());
            }
        }

        const TString distanceTableBlob = BuildPqDistanceTable(residual, codebook, m, nbits);
        const TVector<float> distanceTable = DeserializeFloatVector(distanceTableBlob);
        UNIT_ASSERT_VALUES_EQUAL(distanceTable.size(), static_cast<size_t>(m) * codeCount);

        for (ui32 mIdx = 0; mIdx < m; ++mIdx) {
            const float* subResidual = residualValues.data() + mIdx * subDim;
            for (ui32 codeIdx = 0; codeIdx < codeCount; ++codeIdx) {
                const TVector<float> centroidValues =
                    DeserializeFloatVector(codebookStorage[mIdx][codeIdx]);
                UNIT_ASSERT_VALUES_EQUAL(centroidValues.size(), subDim);
                const float expected = SquaredEuclidean(subResidual, centroidValues.data(), subDim);
                AssertNear(expected, distanceTable[mIdx * codeCount + codeIdx], 1e-4f,
                    TStringBuilder() << "m=" << mIdx << " code=" << codeIdx);
            }
        }
    }

    Y_UNIT_TEST(BuildPqDistanceTable_NonDivisibleDimAsserts) {
        const TString residual = SerializeFloatVector({1.f, 2.f, 3.f});
        const TString centroid = SerializeFloatVector({0.f, 0.f, 0.f});
        const TVector<TVector<TStringBuf>> codebook = {{centroid, centroid}};
        UNIT_ASSERT_EXCEPTION(BuildPqDistanceTable(residual, codebook, /*m=*/2, /*nbits=*/1), yexception);
    }

    Y_UNIT_TEST(ComputePqDistance_Trivial) {
        // distanceTable laid out as M=3 sub-spaces by 2^nbits=256 entries each.
        // Pick codes that select known offsets in the table and check the sum.
        constexpr ui32 m = 3;
        constexpr ui32 nbits = 8;
        constexpr ui32 codeCount = 1u << nbits;

        TVector<float> table(static_cast<size_t>(m) * codeCount, 0.f);
        // Cherry-pick three entries to be summed.
        table[0u * codeCount + 7]   = 1.5f;
        table[1u * codeCount + 200] = 2.25f;
        table[2u * codeCount + 13]  = 0.5f;
        const TString tableBlob = SerializeFloatVector(table);

        const TVector<ui16> codes = {ui8{7}, ui8{200}, ui8{13}};
        const TString codesBlob = NPackedNBitVector::Serialize<ui8>(codes, nbits);

        const double actual = ComputePqDistance(tableBlob, codesBlob, m, nbits);
        AssertNear(1.5f + 2.25f + 0.5f, static_cast<float>(actual), 1e-6f, "trivial sum");
    }

    Y_UNIT_TEST(ComputePqDistance_RoundtripWithBuildPqDistanceTable) {
        // End-to-end consistency: distanceTable from BuildPqDistanceTable, plus
        // codes selected by the same nearest-centroid rule, must reproduce the
        // exact partial-sum L2 distance computed directly from the codebook.
        constexpr ui32 m = 4;
        constexpr ui32 nbits = 8;
        constexpr ui32 codeCount = 1u << nbits;
        constexpr ui32 subDim = 3;
        constexpr ui32 totalDim = m * subDim;

        TReallyFastRng32 rng(123);

        TVector<float> residualValues(totalDim);
        for (auto& v : residualValues) {
            v = rng.GenRandReal2() * 2.f - 1.f;
        }
        const TString residual = SerializeFloatVector(residualValues);

        // For determinism and to exercise the full code range, use a smaller
        // effective code set (N <= codeCount) and pad the rest with copies of
        // the first centroid; this keeps the distanceTable shape consistent.
        constexpr ui32 effectiveCodes = 16;

        TVector<TVector<TVector<float>>> centroidValues(m,
            TVector<TVector<float>>(codeCount, TVector<float>(subDim, 0.f)));
        TVector<TVector<TString>> codebookStorage(m);
        TVector<TVector<TStringBuf>> codebook(m);
        for (ui32 mIdx = 0; mIdx < m; ++mIdx) {
            codebookStorage[mIdx].reserve(codeCount);
            codebook[mIdx].reserve(codeCount);
            for (ui32 codeIdx = 0; codeIdx < codeCount; ++codeIdx) {
                if (codeIdx < effectiveCodes) {
                    for (auto& v : centroidValues[mIdx][codeIdx]) {
                        v = rng.GenRandReal2() * 2.f - 1.f;
                    }
                } else {
                    centroidValues[mIdx][codeIdx] = centroidValues[mIdx][0];
                }
                codebookStorage[mIdx].push_back(SerializeFloatVector(centroidValues[mIdx][codeIdx]));
                codebook[mIdx].push_back(codebookStorage[mIdx].back());
            }
        }

        const TString distanceTable = BuildPqDistanceTable(residual, codebook, m, nbits);

        // Build a sample vector and quantize it manually.
        TVector<float> sampleValues(totalDim);
        for (auto& v : sampleValues) {
            v = rng.GenRandReal2() * 2.f - 1.f;
        }

        TVector<ui8> codes(m);
        for (ui32 mIdx = 0; mIdx < m; ++mIdx) {
            const float* sub = sampleValues.data() + mIdx * subDim;
            float bestDist = std::numeric_limits<float>::max();
            ui32 bestCode = 0;
            for (ui32 codeIdx = 0; codeIdx < effectiveCodes; ++codeIdx) {
                const float dist = SquaredEuclidean(sub, centroidValues[mIdx][codeIdx].data(), subDim);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCode = codeIdx;
                }
            }
            codes[mIdx] = static_cast<ui8>(bestCode);
        }

        const TString codesBlob = NPackedNBitVector::Serialize<ui8>(codes, nbits);
        const double actual = ComputePqDistance(distanceTable, codesBlob, m, nbits);

        // Expected: sum over sub-spaces of squared L2 between the residual sub-vector
        // and the centroid selected by the same code.
        double expected = 0.0;
        for (ui32 mIdx = 0; mIdx < m; ++mIdx) {
            const float* subResidual = residualValues.data() + mIdx * subDim;
            expected += SquaredEuclidean(subResidual, centroidValues[mIdx][codes[mIdx]].data(), subDim);
        }

        AssertNear(static_cast<float>(expected), static_cast<float>(actual), 1e-3f,
            "ComputePqDistance must agree with manual partial-sum L2");
    }

    Y_UNIT_TEST(ComputePqDistance_RejectsBadInput) {
        constexpr ui32 m = 2;
        constexpr ui32 nbits = 8;
        constexpr ui32 codeCount = 1u << nbits;

        const TString tableBlob = SerializeFloatVector(TVector<float>(static_cast<size_t>(m) * codeCount, 0.f));
        const TVector<ui8> codes2(2, 0u);
        const TVector<ui8> codes3(3, 0u);
        const TString codesBlob = NPackedNBitVector::Serialize<ui8>(codes2, nbits);

        // Wrong nbits: only nbits == 8 is currently supported.
        UNIT_ASSERT_EXCEPTION(ComputePqDistance(tableBlob, codesBlob, m, /*nbits=*/4), yexception);

        // Mismatch between distanceTable size and (m, nbits).
        UNIT_ASSERT_EXCEPTION(ComputePqDistance(tableBlob, codesBlob, /*m=*/3, nbits), yexception);

        // Codes count mismatch.
        const TString codesBlob3 = NPackedNBitVector::Serialize<ui8>(codes3, nbits);
        UNIT_ASSERT_EXCEPTION(ComputePqDistance(tableBlob, codesBlob3, m, nbits), yexception);
    }

    Y_UNIT_TEST(BuildPqDistanceTable_SparseCodebookPadsWithInf) {
        // M=2 sub-spaces, each of dim 2; nbits=2 -> codeCount=4.
        // codebook[0] has only 2 of 4 centroids; codebook[1] has only 1 of 4.
        // residual = [1.0, 2.0,  3.0, 4.0]
        // Expected: real entries hold finite squared L2; missing slots = +Inf.
        const TString residual = SerializeFloatVector({1.f, 2.f, 3.f, 4.f});

        const TString c00 = SerializeFloatVector({0.f, 0.f}); // (1,2) -> 5
        const TString c01 = SerializeFloatVector({1.f, 2.f}); // (1,2) -> 0
        const TString c10 = SerializeFloatVector({3.f, 4.f}); // (3,4) -> 0

        const TVector<TVector<TStringBuf>> codebook = {
            {c00, c01},
            {c10},
        };

        const TString distanceTableBlob = BuildPqDistanceTable(residual, codebook, /*m=*/2, /*nbits=*/2);
        const TVector<float> distanceTable = DeserializeFloatVector(distanceTableBlob);

        UNIT_ASSERT_VALUES_EQUAL(distanceTable.size(), 8u);
        AssertNear(5.f, distanceTable[0], 1e-4f, "distanceTable[0,0]");
        AssertNear(0.f, distanceTable[1], 1e-4f, "distanceTable[0,1]");
        UNIT_ASSERT_C(std::isinf(distanceTable[2]) && distanceTable[2] > 0,
            "distanceTable[0,2] must be +Inf, got " << distanceTable[2]);
        UNIT_ASSERT_C(std::isinf(distanceTable[3]) && distanceTable[3] > 0,
            "distanceTable[0,3] must be +Inf, got " << distanceTable[3]);
        AssertNear(0.f, distanceTable[4], 1e-4f, "distanceTable[1,0]");
        for (size_t i = 5; i < 8; ++i) {
            UNIT_ASSERT_C(std::isinf(distanceTable[i]) && distanceTable[i] > 0,
                "distanceTable[1," << (i - 4) << "] must be +Inf, got " << distanceTable[i]);
        }
    }

    Y_UNIT_TEST(BuildPqDistanceTable_FullyEmptyCodebookSegment) {
        // codebook[mIdx].empty() is allowed: the entire segment is +Inf-filled.
        // No assertion — codes never reach an empty segment in normal operation,
        // and we want a deterministic safety net rather than a crash.
        const TString residual = SerializeFloatVector({1.f, 2.f, 3.f, 4.f});
        const TString c0 = SerializeFloatVector({0.f, 0.f});

        const TVector<TVector<TStringBuf>> codebook = {
            {c0, c0},
            {}, // intentionally empty
        };

        const TString distanceTableBlob = BuildPqDistanceTable(residual, codebook, /*m=*/2, /*nbits=*/1);
        const TVector<float> distanceTable = DeserializeFloatVector(distanceTableBlob);

        UNIT_ASSERT_VALUES_EQUAL(distanceTable.size(), 4u);
        // Segment 0: two real entries.
        UNIT_ASSERT_C(std::isfinite(distanceTable[0]), "distanceTable[0,0] must be finite");
        UNIT_ASSERT_C(std::isfinite(distanceTable[1]), "distanceTable[0,1] must be finite");
        // Segment 1: fully padded.
        UNIT_ASSERT_C(std::isinf(distanceTable[2]) && distanceTable[2] > 0,
            "distanceTable[1,0] must be +Inf");
        UNIT_ASSERT_C(std::isinf(distanceTable[3]) && distanceTable[3] > 0,
            "distanceTable[1,1] must be +Inf");
    }

    Y_UNIT_TEST(ComputePqDistance_PropagatesInfFromSparseSlot) {
        // Sentinel semantics in the hot path: a code pointing into a +Inf slot
        // makes the whole sum +Inf; a code pointing into a finite slot returns
        // a finite sum even if other slots in the same segment are +Inf.
        constexpr ui32 m = 2;
        constexpr ui32 nbits = 8;
        constexpr ui32 codeCount = 1u << nbits;
        constexpr float kInf = std::numeric_limits<float>::infinity();

        TVector<float> table(static_cast<size_t>(m) * codeCount, kInf);
        // Reachable, finite cells.
        table[0u * codeCount + 5]  = 0.25f;
        table[1u * codeCount + 17] = 0.75f;
        const TString tableBlob = SerializeFloatVector(table);

        // Codes hitting only the finite cells -> finite sum.
        const TVector<ui8> finiteCodes = {ui8{5}, ui8{17}};
        const TString finiteCodesBlob = NPackedNBitVector::Serialize<ui8>(finiteCodes, nbits);
        const double finiteResult = ComputePqDistance(tableBlob, finiteCodesBlob, m, nbits);
        AssertNear(1.0f, static_cast<float>(finiteResult), 1e-6f, "finite-codes path");

        // First code hits a +Inf cell -> +Inf result.
        const TVector<ui8> infCodes = {ui8{42}, ui8{17}};
        const TString infCodesBlob = NPackedNBitVector::Serialize<ui8>(infCodes, nbits);
        const double infResult = ComputePqDistance(tableBlob, infCodesBlob, m, nbits);
        UNIT_ASSERT_C(std::isinf(infResult) && infResult > 0,
            "expected +Inf when a code points into a sentinel slot, got " << infResult);
    }
}

} // namespace NKikimr::NIvfPq
