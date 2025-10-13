package ai.koog.rag.vector

import ai.koog.embeddings.base.Vector
import ai.koog.rag.base.chunking.DocumentChunk
import ai.koog.rag.base.chunking.ParagraphChunker
import ai.koog.rag.base.files.TextRange
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import org.testcontainers.containers.PostgreSQLContainer
import org.testcontainers.utility.DockerImageName
import java.sql.Connection
import java.sql.DriverManager
import kotlin.test.assertEquals

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PgVectorStorageTest {

    companion object {
        // Configures the PostgreSQLContainer to use an image that contains a version of Postgres
        // with the pgvector extension already installed and enabled.
        class KPostgresContainer : PostgreSQLContainer<KPostgresContainer>(
            DockerImageName.parse("pgvector/pgvector:0.8.1-pg18-trixie")
        )
    }

    private lateinit var postgres: KPostgresContainer
    private lateinit var connProvider: () -> Connection

    @BeforeAll
    fun startContainer() {
        postgres = KPostgresContainer()
            .apply {
                withDatabaseName("testdb")
                withUsername("testuser")
                withPassword("testpw")
                start()
            }
        connProvider = {
            DriverManager.getConnection(
                postgres.jdbcUrl, postgres.username, postgres.password
            )
        }
    }

    @AfterAll
    fun stopContainer() {
        postgres.stop()
    }

    @Test
    fun `FLOAT factory with default operator builds correct PgVectorStorage`() {
        @Serializable data class Dummy(val x: Int)

        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = Dummy.serializer(),
            tableName = "test_factory_float_default"
        )
        assertEquals(VectorStorageType.FLOAT_VECTOR, storage.getVectorType)
        assertEquals(setOf(VectorDistanceOperator.L2), storage.getDistanceOperators)
    }

    @Test
    fun `FLOAT factory with named params and custom operator set`() {
        @Serializable data class Dummy(val x: Int)
        val allOps = setOf(
            VectorDistanceOperator.L2,
            VectorDistanceOperator.L1,
            VectorDistanceOperator.COSINE,
            VectorDistanceOperator.DOT_PRODUCT
        )

        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 3,
            serializer = Dummy.serializer(),
            json = Json,
            tableName = "test_factory_float_custom_ops",
            distanceOperators = allOps
        )
        assertEquals(VectorStorageType.FLOAT_VECTOR, storage.getVectorType)
        assertEquals(allOps, storage.getDistanceOperators)
    }

    @Test
    fun `BIT factory default operator builds correct PgVectorStorage`() {
        @Serializable data class Dummy(val y: String)
        val bitStorage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 8,
            serializer = Dummy.serializer(),
            tableName = "test_factory_bit_default"
        )
        assertEquals(VectorStorageType.BIT_VECTOR, bitStorage.getVectorType)
        assertEquals(setOf(VectorDistanceOperator.HAMMING), bitStorage.getDistanceOperators)
    }

    @Test
    fun `BIT factory with custom operator set all allowed ops`() {
        @Serializable data class Dummy(val y: String)
        val ops = setOf(VectorDistanceOperator.HAMMING, VectorDistanceOperator.JACCARD)
        val bitStorage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 4,
            serializer = Dummy.serializer(),
            tableName = "test_factory_bit_custom_ops",
            distanceOperators = ops
        )
        assertEquals(VectorStorageType.BIT_VECTOR, bitStorage.getVectorType)
        assertEquals(ops, bitStorage.getDistanceOperators)
    }

    @Test
    fun `FLOAT factory throws with invalid operators`() {
        @Serializable data class Dummy(val z: Int)
        val forbiddenOps = setOf(VectorDistanceOperator.HAMMING, VectorDistanceOperator.JACCARD)
        val ex = assertThrows<IllegalArgumentException> {
            PgVectorStorage.floatVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 2,
                serializer = Dummy.serializer(),
                tableName = "test_factory_float_invalid_ops",
                distanceOperators = forbiddenOps
            )
        }
        assertTrue("Operators" in ex.message!!)
    }

    @Test
    fun `BIT factory throws with non-binary operators`() {
        @Serializable data class Dummy(val z: Int)
        val forbiddenOps = setOf(VectorDistanceOperator.L2, VectorDistanceOperator.L1, VectorDistanceOperator.COSINE)
        val ex = assertThrows<IllegalArgumentException> {
            PgVectorStorage.bitVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 3,
                serializer = Dummy.serializer(),
                tableName = "test_factory_bit_invalid_ops",
                distanceOperators = forbiddenOps
            )
        }
        assertTrue("Operators" in ex.message!!)
    }

    @Test
    fun `float vector storage fails with vector dimension too large for HNSW index`() {
        @Serializable data class Dummy(val d: Int)
        val ex = assertThrows<IllegalArgumentException> {
            PgVectorStorage.floatVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 2001, // Exceeds the 2000-dim HNSW limit
                serializer = Dummy.serializer(),
                tableName = "test_float_too_many_dims"
            )
        }
        val expectedMessage =
            "PgVector HNSW index supports at most 2000 dimensions for type FLOAT_VECTOR (requested 2001); see https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw"
        assertEquals(expectedMessage, ex.message)
    }


    @Test
    fun `bit vector storage fails with vector dimension too large for HNSW index`() {
        @Serializable data class Dummy(val d: Int)
        val ex = assertThrows<IllegalArgumentException> {
            PgVectorStorage.bitVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 64001, // Exceeds the 64,000-dim HNSW limit
                serializer = Dummy.serializer(),
                tableName = "test_bit_too_many_dims"
            )
        }
        val expectedMessage =
            "PgVector HNSW index supports at most 64000 dimensions for type BIT_VECTOR (requested 64001); see https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw"
        assertEquals(expectedMessage, ex.message)
    }

    @Test
    fun `float vector storage succeeds with vector dimension at 2000 (HNSW limit)`() {
        @Serializable data class Dummy(val d: Int)
        // This should NOT throw
        try {
            PgVectorStorage.floatVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 2000, // Exactly at the 2000-dim HNSW limit
                serializer = Dummy.serializer(),
                tableName = "test_float_at_limit"
            )
        } catch (e: Exception) {
            fail("Should allow dimension of 2000 for FLOAT_VECTOR but threw: ${e.message}")
        }
    }

    @Test
    fun `bit vector storage succeeds with vector dimension at 64000 (HNSW limit)`() {
        @Serializable data class Dummy(val d: Int)
        // This should NOT throw
        try {
            PgVectorStorage.bitVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 64000, // Exactly at the 64,000-dim HNSW limit
                serializer = Dummy.serializer(),
                tableName = "test_bit_at_limit"
            )
        } catch (e: Exception) {
            fail("Should allow dimension of 64000 for BIT_VECTOR but threw: ${e.message}")
        }
    }

    @Test
    fun `store works with valid float vector`() = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 3,
            serializer = Dummy.serializer(),
            tableName = "test_store_float_good"
        )
        val doc = DocumentWithMetadata(content = Dummy(5))
        val vec = Vector(listOf(1.0, 2.0, 3.0))
        val id = storage.store(doc, vec)
        assertNotNull(id)
    }

    @Test
    fun `store fails for invalid float vector dimension`() = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = Dummy.serializer(),
            tableName = "test_store_float_dim_bad"
        )
        val doc = DocumentWithMetadata(content = Dummy(5))
        val vec = Vector(listOf(1.0, 2.0, 3.0)) // Wrong size
        val ex = assertThrows<IllegalArgumentException> {
            storage.store(doc, vec)
        }
        assertTrue("Vector size" in ex.message!!)
    }

    @Test
    fun `store works with bit vector containing only 0 and 1`() = runBlocking {
        @Serializable data class Dummy(val y: Int)
        val storage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 4,
            serializer = Dummy.serializer(),
            tableName = "test_store_bit_valid"
        )
        val doc = DocumentWithMetadata(content = Dummy(7))
        val vec = Vector(listOf(1.0, 0.0, 1.0, 1.0))
        val id = storage.store(doc, vec)
        assertNotNull(id)
    }

    @Test
    fun `store fails with bit vector that has non-binary value`() = runBlocking {
        @Serializable data class Dummy(val y: Int)
        val storage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 3,
            serializer = Dummy.serializer(),
            tableName = "test_store_bit_invalid"
        )
        val doc = DocumentWithMetadata(content = Dummy(8))
        val badVec = Vector(listOf(1.0, 0.5, 0.0)) // 0.5 is NOT allowed
        val ex = assertThrows<IllegalArgumentException> {
            storage.store(doc, badVec)
        }
        assertTrue("BIT_VECTOR storage only supports vector values 0.0 or 1.0" in ex.message!!)
    }

    @Test
    fun `store fails with bit vector having wrong dimension`() = runBlocking {
        @Serializable data class Dummy(val z: String)
        val storage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = Dummy.serializer(),
            tableName = "test_store_bit_wrongdim"
        )
        val doc = DocumentWithMetadata(content = Dummy("dim"))
        val wrongDimVec = Vector(listOf(0.0, 1.0, 0.0)) // 3 values, expected 2
        val ex = assertThrows<IllegalArgumentException> {
            storage.store(doc, wrongDimVec)
        }
        assertTrue("Vector size" in ex.message!!)
    }

    @Test
    fun `store and retrieve a document with metadata`() = runBlocking {
        @Serializable
        data class MyDoc(val text: String, val id: Int)
        val json = Json
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 3,
            serializer = MyDoc.serializer(),
            json = json,
            tableName = "test_vector_store",
            distanceOperators = setOf(VectorDistanceOperator.L2)
        )
        val myDoc = MyDoc(text = "Doc test 1", id = 101)
        val doc = DocumentWithMetadata(
            content = myDoc,
            documentType = "doc_test_type",
            source = "test-doc-source",
            tags = listOf("independent", "metadata"),
            metadata = mapOf("foo" to JsonPrimitive("bar"))
        )
        val vec = Vector(listOf(0.11, 0.22, 0.33))
        val id = storage.store(doc, vec)
        assertNotNull(id)
        val loadedDoc = storage.read(id)
        assertNotNull(loadedDoc)
        assertEquals(myDoc, loadedDoc!!.content)
        assertEquals(doc.documentType, loadedDoc.documentType)
        assertEquals(doc.tags, loadedDoc.tags)
        assertEquals(doc.metadata, loadedDoc.metadata)
        assertEquals(doc.source, loadedDoc.source)
    }

    @Test
    fun `store and retrieve large vector with expected float precision loss`() = runBlocking {
        @Serializable
        data class VecDoc(val description: String)
        val json = Json
        val vectorSize = 1024
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = vectorSize,
            serializer = VecDoc.serializer(),
            json = json,
            tableName = "test_vector_store_large_${vectorSize}",
            distanceOperators = setOf(VectorDistanceOperator.L2),
        )
        val vecDoc = VecDoc(description = "Large vector precision test")
        val largeVec = Vector(
            // Generate a vector with values ranging from 0 to 1023.5 in steps of 0.5
            List(vectorSize) { it * 0.5 }
        )
        val doc = DocumentWithMetadata(
            content = vecDoc
        )
        val id = storage.store(doc, largeVec)
        assertNotNull(id)
        /*
         * Verify that the large vector embedding is correctly stored and retrieved.
         *
         * We convert our local list of doubles to floats before storing because pgvector
         * expects the underlying data as float32, not float64 (double).
         * When retrieving, we convert the float[] back to a List<Double>, which can introduce
         * tiny floating-point precision losses. This is why we compare with a tolerance.
         */
        val roundTrippedVec = storage.getPayload(id)
        assertNotNull(roundTrippedVec)
        assertEquals(largeVec.values.size, roundTrippedVec!!.values.size)
        /*
         * For each value, assert they are "close enough" (within 1e-6) to account for
         * any small rounding errors in floating-point math and conversion.
         *
         * Assert that 'given' and 'stored' values are equal up to a tolerance of 1e-6,
         * handling minor differences caused by storing as float in pgvector and reading
         * back as double in Kotlin.
         */
        largeVec.values.zip(roundTrippedVec.values).forEachIndexed { idx, (given, stored) ->
            assertEquals(given, stored, 1e-6, "Mismatch at index $idx")
        }
    }

    @ParameterizedTest
    @EnumSource(
        value = VectorDistanceOperator::class,
        names = ["L2", "DOT_PRODUCT", "L1", "COSINE"]
    )
    fun `topKSimilarDocumentsWithOperator finds nearest neighbors correctly for each operator`(
        operator: VectorDistanceOperator
    ) = runBlocking {
        @Serializable
        data class Doc(val label: String)
        val json = Json
        val vectorSize = 3
        val tableName = "vector_topk_test_${operator.name.lowercase()}"
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = vectorSize,
            serializer = Doc.serializer(),
            json = json,
            tableName = tableName,
            distanceOperators = setOf(operator)
        )
        val docs = listOf(
            DocumentWithMetadata(content = Doc("A")), // [0.0, 0.0, 0.0]
            DocumentWithMetadata(content = Doc("B")), // [1.0, 1.0, 1.0]
            DocumentWithMetadata(content = Doc("C")), // [2.0, 2.0, 2.0]
            DocumentWithMetadata(content = Doc("D")), // [10.0, 10.0, 10.0]
            DocumentWithMetadata(content = Doc("E"))  // [100.0, 100.0, 100.0]
        )
        val vectors = listOf(
            Vector(listOf(0.0, 0.0, 0.0)),
            Vector(listOf(1.0, 1.0, 1.0)),
            Vector(listOf(2.0, 2.0, 2.0)),
            Vector(listOf(10.0, 10.0, 10.0)),
            Vector(listOf(100.0, 100.0, 100.0))
        )
        vectors.forEach {
            assert(it.values.size == vectorSize) { "Vector has wrong size: ${it.values.size}" }
        }

        val ids = docs.zip(vectors).map { (doc, vec) -> storage.store(doc, vec) }
        assertEquals(docs.size, ids.size)
        val allDocsWithPayload = storage.allDocumentsWithPayload().toList()
        assertEquals(docs.size, allDocsWithPayload.size)
        val allLabels = allDocsWithPayload.map { it.document.content.label }
        assertTrue(docs.map { it.content.label }.all { label -> label in allLabels })

        val queryVec = when (operator) {
            VectorDistanceOperator.COSINE, VectorDistanceOperator.DOT_PRODUCT -> Vector(listOf(1.0, 1.0, 1.0))
            else -> Vector(listOf(0.0, 0.0, 0.0))
        }
        val topK = 3
        val results: List<Pair<DocumentWithMetadata<Doc>, Double>> = storage
            .topKSimilarDocumentsWithOperator(queryVec, topK, operator).toList()
        val labels = results.map { it.first.content.label }
        val distances = results.map { it.second }

        when (operator) {
            VectorDistanceOperator.COSINE -> {
                assertEquals(topK, labels.size)
                assertTrue(distances.zipWithNext { a, b -> a <= b }.all { it }, "Distances not monotonic for $operator")
                // Should be one of B, C, D, or E (never A)
                assertTrue(labels.first() in listOf("B", "C", "D", "E"), "First label for COSINE was ${labels.first()}, but should be a positive-direction vector")
                // Check that all cosine distances in the topK are equal
                val expected = distances.first()
                assertTrue(distances.all { it == expected }, "All distances for identical direction should be equal for cosine")
            }
            VectorDistanceOperator.DOT_PRODUCT -> {
                // For DOT_PRODUCT, results are ordered by most negative value (lowest) first,
                // so they are monotonically ascending, e.g., -300, -30, -6.
                assertEquals(topK, labels.size)
                assertTrue(distances.zipWithNext { a, b -> a <= b }.all { it }, "Distances not monotonic (should be ascending) for $operator")
                // Expect "E" or "D" as first, since these have the highest (most negative) dot product with a positive-valued query.
                assertTrue(labels.first() in listOf("E", "D"))
            }
            else -> {
                // L2, L1: "A" is always closest for query (0,0,0)
                assertEquals("A", labels.first())
                assertEquals(topK, results.size)
                assertTrue(distances.zipWithNext { a, b -> a <= b }.all { it }, "Distances not monotonic for $operator")
            }
        }
    }

    @Test
    fun `hamming and jaccard operators on binary vectors`() = runBlocking {
        @Serializable
        data class Doc(val label: String)
        val json = Json
        val vectorSize = 4
        val tableName = "vector_binary_test"
        val bitVectorStorage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = vectorSize,
            serializer = Doc.serializer(),
            json = json,
            tableName = tableName,
            distanceOperators = setOf(VectorDistanceOperator.HAMMING, VectorDistanceOperator.JACCARD)
        )
        val docs = listOf(
            DocumentWithMetadata(content = Doc("A")), // [0, 0, 0, 0]
            DocumentWithMetadata(content = Doc("B")), // [0, 1, 0, 1]
            DocumentWithMetadata(content = Doc("C")), // [1, 1, 1, 1]
            DocumentWithMetadata(content = Doc("D"))  // [1, 0, 1, 0]
        )
        val vectors = listOf(
            Vector(listOf(0.0, 0.0, 0.0, 0.0)),
            Vector(listOf(0.0, 1.0, 0.0, 1.0)),
            Vector(listOf(1.0, 1.0, 1.0, 1.0)),
            Vector(listOf(1.0, 0.0, 1.0, 0.0))
        )
        // Store all documents
        docs.zip(vectors).forEach { (doc, vec) -> bitVectorStorage.store(doc, vec) }
        val queryVec = Vector(listOf(0.0, 1.0, 0.0, 1.0))

        // HAMMING test (ordered by increasing Hamming distance)
        val hammingResults = bitVectorStorage.topKSimilarDocumentsWithOperator(
            queryVector = queryVec,
            topK = 4,
            operator = VectorDistanceOperator.HAMMING
        ).toList()
        val hammingLabels = hammingResults.map { it.first.content.label }
        val hammingDistances = hammingResults.map { it.second }
        // Closest: B (distance 0, exact match)
        assertEquals("B", hammingLabels.first())
        assertEquals(0.0, hammingDistances.first())
        // Next two are tied at distance 2: A and C, order doesn't matter, and D is distance 4
        val expectedTies = setOf("A", "C")
        assertTrue(setOf(hammingLabels[1], hammingLabels[2]) == expectedTies)
        assertEquals(setOf(2.0, 2.0), setOf(hammingDistances[1], hammingDistances[2]))
        assertEquals("D", hammingLabels.last())
        assertEquals(4.0, hammingDistances.last())

        // JACCARD test (ordered by increasing Jaccard distance)
        val jaccardResults = bitVectorStorage.topKSimilarDocumentsWithOperator(
            queryVector = queryVec,
            topK = 4,
            operator = VectorDistanceOperator.JACCARD
        ).toList()
        val jaccardLabels = jaccardResults.map { it.first.content.label }
        val jaccardDistances = jaccardResults.map { it.second }
        // Closest: B (distance 0.0, exact match)
        assertEquals("B", jaccardLabels.first())
        assertEquals(0.0, jaccardDistances.first())
        // All Jaccard distances should be in [0, 1]
        assertTrue(jaccardDistances.all { it in 0.0..1.0 })
    }

    @Test
    fun `throws if zero query vector is used with COSINE operator`() = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = Dummy.serializer(),
            tableName = "test_topk_unsupported_ops_l2only",
            distanceOperators = setOf(VectorDistanceOperator.COSINE)
        )
        val ex = assertThrows<IllegalArgumentException> {
            storage.topKSimilarDocumentsWithOperator(
                queryVector = Vector(listOf(0.0, 0.0)), // Zero query vector
                topK = 5,
                operator = VectorDistanceOperator.COSINE
            ).toList()
        }
        assertTrue("zero query vector" in ex.message!!)
    }

    @Test
    fun `throws if operator is not supported - BIT storage with HAMMING only, JACCARD fails`() = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val storage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 4,
            serializer = Dummy.serializer(),
            tableName = "test_topk_unsupported_ops_hammingonly",
            distanceOperators = setOf(VectorDistanceOperator.HAMMING)
        )
        val ex = assertThrows<IllegalArgumentException> {
            storage.topKSimilarDocumentsWithOperator(
                queryVector = Vector(listOf(1.0, 0.0, 0.0, 1.0)),
                topK = 5,
                operator = VectorDistanceOperator.JACCARD // Not configured in storage
            ).toList()
        }
        assertTrue("not supported" in ex.message!!)
    }

    @Test
    fun `throws if operator is not supported - FLOAT storage with L2 only, DOT_PRODUCT fails`() = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 3,
            serializer = Dummy.serializer(),
            tableName = "test_topk_unsupported_ops_dotproduct",
            distanceOperators = setOf(VectorDistanceOperator.L2)
        )
        val ex = assertThrows<IllegalArgumentException> {
            storage.topKSimilarDocumentsWithOperator(
                queryVector = Vector(listOf(1.0, 2.0, 3.0)),
                topK = 5,
                operator = VectorDistanceOperator.DOT_PRODUCT // Not configured in storage
            ).toList()
        }
        assertTrue("not supported" in ex.message!!)
    }

    @ParameterizedTest
    @EnumSource(VectorDistanceOperator::class)
    fun `topKSimilarDocumentsWithNormalizedSimilarity handles operator support and validation`(operator: VectorDistanceOperator) = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val supported = setOf(
            VectorDistanceOperator.L2, VectorDistanceOperator.L1,
            VectorDistanceOperator.COSINE, VectorDistanceOperator.HAMMING,
            VectorDistanceOperator.JACCARD
        )
        val isBitOp = operator == VectorDistanceOperator.HAMMING || operator == VectorDistanceOperator.JACCARD
        val storage = if (isBitOp)
            PgVectorStorage.bitVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 2,
                serializer = Dummy.serializer(),
                tableName = "test_normalized_sim_bit_${operator.name.lowercase()}",
                distanceOperators = setOf(operator)
            )
        else
            PgVectorStorage.floatVectorStorage(
                connectionProvider = connProvider,
                vectorDimension = 2,
                serializer = Dummy.serializer(),
                tableName = "test_normalized_sim_float_${operator.name.lowercase()}",
                distanceOperators = setOf(operator)
            )

        // Store multiple documents/vectors to test actual nearest-neighbor semantics
        val docs = listOf(
            DocumentWithMetadata(content = Dummy(1)),
            DocumentWithMetadata(content = Dummy(2)),
            DocumentWithMetadata(content = Dummy(3))
        )
        val vectors = if (isBitOp) {
            listOf(
                Vector(listOf(1.0, 0.0)),
                Vector(listOf(0.0, 1.0)),
                Vector(listOf(1.0, 1.0))
            )
        } else {
            listOf(
                Vector(listOf(1.0, 2.0)),
                Vector(listOf(3.0, 4.0)),
                Vector(listOf(-1.0, -2.0))
            )
        }
        docs.zip(vectors).forEach { (doc, vec) -> storage.store(doc, vec) }

        val query =
            if (operator == VectorDistanceOperator.COSINE) Vector(listOf(0.0, 0.0))
            else Vector(listOf(1.0, 2.0))

        if (operator == VectorDistanceOperator.DOT_PRODUCT) {
            val ex = assertThrows<IllegalArgumentException> {
                storage.topKSimilarDocumentsWithNormalizedSimilarity(query, 1, operator).toList()
            }
            assertTrue("does not support DOT_PRODUCT" in ex.message!!)
        } else if (operator == VectorDistanceOperator.COSINE && query.values.all { it == 0.0 }) {
            val ex = assertThrows<IllegalArgumentException> {
                storage.topKSimilarDocumentsWithNormalizedSimilarity(query, 1, operator).toList()
            }
            assertTrue("zero vector" in ex.message!!)
        } else if (operator in supported) {
            // Should work for supported operators (with nonzero vector for COSINE)
            if (operator == VectorDistanceOperator.COSINE) {
                // Try again with a nonzero vector for a passing case
                val nonzeroQuery = Vector(listOf(1.0, 1.0))
                val result = storage.topKSimilarDocumentsWithNormalizedSimilarity(nonzeroQuery, 1, operator).toList()
                assertTrue(result.isNotEmpty())
                assertTrue(result.first().second in 0.0..1.0)
            } else {
                val result = storage.topKSimilarDocumentsWithNormalizedSimilarity(query, 1, operator).toList()
                assertTrue(result.isNotEmpty())
                assertTrue(result.first().second in 0.0..1.0)
            }
        } else {
            // If the operator is not in the storage's allowed set, should throw
            val ex = assertThrows<IllegalArgumentException> {
                storage.topKSimilarDocumentsWithNormalizedSimilarity(query, 1, operator).toList()
            }
            assertTrue("not supported" in ex.message!!)
        }
    }

    @Test
    fun `read returns expected document and vector - float vector storage`() = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = Dummy.serializer(),
            tableName = "test_read_float"
        )
        val doc = DocumentWithMetadata(content = Dummy(42))
        val vec = Vector(listOf(0.2, 0.9))
        val id = storage.store(doc, vec)
        val loadedDoc = storage.read(id)
        assertNotNull(loadedDoc)
        assertEquals(doc.content, loadedDoc!!.content)
        val loadedVec = storage.getPayload(id)
        assertNotNull(loadedVec)
        vec.values.zip(loadedVec!!.values).forEach { (expected, actual) ->
            assertEquals(expected, actual, 1e-6) // Accounting for float precision loss
        }
        val loadedWithPayload = storage.readWithPayload(id)
        assertNotNull(loadedWithPayload)
        loadedWithPayload!!.payload.values.zip(vec.values).forEach { (actual, expected) ->
            assertEquals(expected, actual, 1e-6) // Accounting for float precision loss
        }
    }

    @Test
    fun `read returns expected document and vector - bit vector storage`() = runBlocking {
        @Serializable data class Dummy(val label: String)
        val storage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 4,
            serializer = Dummy.serializer(),
            tableName = "test_read_bit"
        )
        val doc = DocumentWithMetadata(content = Dummy("bitLabel"))
        val vec = Vector(listOf(1.0, 0.0, 1.0, 1.0))
        val id = storage.store(doc, vec)
        val loadedDoc = storage.read(id)
        assertNotNull(loadedDoc)
        assertEquals(doc.content, loadedDoc!!.content)
        val loadedVec = storage.getPayload(id)
        assertNotNull(loadedVec)
        assertEquals(vec.values, loadedVec!!.values)
        val loadedWithPayload = storage.readWithPayload(id)
        assertNotNull(loadedWithPayload)
        assertEquals(doc.content, loadedWithPayload!!.document.content)
        assertEquals(vec.values, loadedWithPayload.payload.values)
    }

    @Test
    fun `read returns null for non-existent document`() = runBlocking {
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = String.serializer(),
            tableName = "test_read_nonexistent"
        )
        val nonExistentId = "00000000-0000-0000-0000-000000000000"
        assertNull(storage.read(nonExistentId))
        assertNull(storage.getPayload(nonExistentId))
        assertNull(storage.readWithPayload(nonExistentId))
    }

    @Test
    fun `delete removes document and its vector - float vector storage`() = runBlocking {
        @Serializable data class Dummy(val x: Int)
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = Dummy.serializer(),
            tableName = "test_delete_float"
        )
        val doc = DocumentWithMetadata(content = Dummy(123))
        val vec = Vector(listOf(0.4, 0.7))
        val id = storage.store(doc, vec)
        assertNotNull(storage.read(id))
        assertNotNull(storage.getPayload(id))
        assertNotNull(storage.readWithPayload(id))
        val deleted = storage.delete(id)
        assertTrue(deleted, "Document should be deleted.")
        assertNull(storage.read(id))
        assertNull(storage.getPayload(id))
        assertNull(storage.readWithPayload(id))
    }

    @Test
    fun `delete removes document and its vector - bit vector storage`() = runBlocking {
        @Serializable data class Dummy(val label: String)
        val storage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 4,
            serializer = Dummy.serializer(),
            tableName = "test_delete_bit"
        )
        val doc = DocumentWithMetadata(content = Dummy("bitdoc"))
        val vec = Vector(listOf(1.0, 0.0, 1.0, 1.0))
        val id = storage.store(doc, vec)
        assertNotNull(storage.read(id))
        assertNotNull(storage.getPayload(id))
        assertNotNull(storage.readWithPayload(id))
        val deleted = storage.delete(id)
        assertTrue(deleted, "Document should be deleted.")
        assertNull(storage.read(id))
        assertNull(storage.getPayload(id))
        assertNull(storage.readWithPayload(id))
    }

    @Test
    fun `delete returns false if document does not exist`() = runBlocking {
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = String.serializer(),
            tableName = "test_delete_missing"
        )
        val nonExistentId = "00000000-0000-0000-0000-000000000000"
        val deleted = storage.delete(nonExistentId)
        assertFalse(deleted, "Should return false when deleting a missing document.")
    }

    @Test
    fun `allDocuments and allDocumentsWithPayload return all for float vector storage`() = runBlocking {
        @Serializable data class Dummy(val d: Int)
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 2,
            serializer = Dummy.serializer(),
            tableName = "test_all_docs_float"
        )
        val docs = listOf(
            DocumentWithMetadata(content = Dummy(11)),
            DocumentWithMetadata(content = Dummy(22)),
            DocumentWithMetadata(content = Dummy(33))
        )
        val vectors = listOf(
            Vector(listOf(0.5, 1.0)),
            Vector(listOf(2.2, -0.6)),
            Vector(listOf(-1.4, 0.3))
        )
        docs.zip(vectors).forEach { (doc, vec) -> storage.store(doc, vec) }
        // Check allDocuments
        val foundDocs = storage.allDocuments().toList()
        assertEquals(docs.size, foundDocs.size)
        docs.forEach { expected ->
            assertTrue(foundDocs.any { it.content == expected.content })
        }
        // Check allDocumentsWithPayload (use tolerance for floats)
        val allPayload = storage.allDocumentsWithPayload().toList()
        assertEquals(docs.size, allPayload.size)
        docs.zip(vectors).forEach { (expectedDoc, expectedVec) ->
            val entry = allPayload.find { it.document.content == expectedDoc.content }
            assertNotNull(entry)
            expectedVec.values.zip(entry!!.payload.values).forEach { (expected, actual) ->
                assertEquals(expected, actual, 1e-6)
            }
        }
    }

    @Test
    fun `allDocuments and allDocumentsWithPayload return all for bit vector storage`() = runBlocking {
        @Serializable data class Dummy(val s: String)
        val storage = PgVectorStorage.bitVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 4,
            serializer = Dummy.serializer(),
            tableName = "test_all_docs_bit"
        )
        val docs = listOf(
            DocumentWithMetadata(content = Dummy("a")),
            DocumentWithMetadata(content = Dummy("b"))
        )
        val vectors = listOf(
            Vector(listOf(0.0, 1.0, 1.0, 0.0)),
            Vector(listOf(1.0, 1.0, 0.0, 1.0))
        )
        docs.zip(vectors).forEach { (doc, vec) -> storage.store(doc, vec) }
        val foundDocs = storage.allDocuments().toList()
        assertEquals(docs.size, foundDocs.size)
        docs.forEach { expected ->
            assertTrue(foundDocs.any { it.content == expected.content })
        }
        val allPayload = storage.allDocumentsWithPayload().toList()
        assertEquals(docs.size, allPayload.size)
        docs.zip(vectors).forEach { (expectedDoc, expectedVec) ->
            val entry = allPayload.find { it.document.content == expectedDoc.content }
            assertNotNull(entry)
            assertEquals(expectedVec.values, entry!!.payload.values)
        }
    }

    @Test
    fun `can store and query document chunks with metadata in PgVectorStorage`() = runBlocking {
        // The original document (with paragraphs)
        val document = "Intro paragraph.\n\nDetail paragraph.\n\nConclusion paragraph."
        val chunker = ParagraphChunker()
        val documentChunks: List<DocumentChunk<String>> = chunker.chunk(document)
        assertTrue(documentChunks.size >= 2, "Should have at least two chunks (paragraphs).")

        // Set up PgVectorStorage for DocumentWithMetadata<DocumentChunk<String>>
        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 3,
            serializer = DocumentChunk.serializer(String.serializer()),
            tableName = "test_chunking_vector_with_metadata"
        )

        // Fake embedding: just use the text length, real code would use an embedder
        fun fakeEmbedding(chunk: DocumentChunk<String>): Vector =
            Vector(listOf(chunk.text.length.toDouble(), 0.0, 0.0))

        // Store all chunks, each wrapped with per-chunk metadata
        val chunkIdPairs = documentChunks.mapIndexed { idx, chunk ->
            val document = DocumentWithMetadata(
                content = chunk,
                documentType = "paragraph-chunk",
                source = "test-doc",
                tags = listOf("auto-chunked", "test"),
                metadata = mapOf("chunkIndex" to JsonPrimitive(idx))
            )
            val id = storage.store(document, fakeEmbedding(chunk))
            id to document
        }
        assertEquals(documentChunks.size, chunkIdPairs.size)

        for ((storedDocumentId, storedDocumentWithMetadata) in chunkIdPairs) {
            val retrievedDocumentWithMetadata = storage.read(storedDocumentId)
            assertNotNull(retrievedDocumentWithMetadata, "read($storedDocumentId) returned null")
            assertEquals(storedDocumentWithMetadata.content.text, retrievedDocumentWithMetadata!!.content.text)
            assertEquals(storedDocumentWithMetadata.documentType, retrievedDocumentWithMetadata.documentType)
            assertEquals(storedDocumentWithMetadata.source, retrievedDocumentWithMetadata.source)
            assertEquals(storedDocumentWithMetadata.tags, retrievedDocumentWithMetadata.tags)
            assertEquals(storedDocumentWithMetadata.metadata, retrievedDocumentWithMetadata.metadata)
        }

        // Search for a chunk containing "Intro"
        val queryDocumentChunk = DocumentChunk(
            parent = document,
            range = TextRange(0, 5),
            text = "Intro"
        )
        val queryEmbedding = fakeEmbedding(queryDocumentChunk)

        val similarDocumentsAndEmbeddings = storage.topKSimilarDocumentsWithOperator(
            queryVector = queryEmbedding,
            topK = 1,
            operator = VectorDistanceOperator.L2
        ).toList()

        assertTrue(similarDocumentsAndEmbeddings.isNotEmpty())
        val (mostSimilarDocument, _) = similarDocumentsAndEmbeddings.first()
        assertTrue(mostSimilarDocument.content.text.contains("Intro"), "Should match intro paragraph chunk")
        assertEquals("paragraph-chunk", mostSimilarDocument.documentType)
        assertEquals("test-doc", mostSimilarDocument.source)
        assertTrue(mostSimilarDocument.tags.contains("auto-chunked"))
    }
}
