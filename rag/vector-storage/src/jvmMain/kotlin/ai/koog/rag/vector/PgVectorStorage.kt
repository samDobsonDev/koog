package ai.koog.rag.vector

import ai.koog.embeddings.base.Vector
import ai.koog.rag.base.DocumentWithPayload
import ai.koog.rag.base.RankedDocument
import com.pgvector.PGbit
import com.pgvector.PGvector
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.sql.Connection
import java.sql.ResultSet
import java.util.UUID
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import org.postgresql.util.PGobject

/**
 * A data class that wraps a document with additional metadata fields commonly needed for storage and retrieval.
 *
 * This class associates a document with metadata such as document type, source, tags, and arbitrary key-value metadata.
 * When used with storage implementations like [PgVectorStorage] these fields are persisted alongside the document content and can be queried or filtered.
 *
 * @param Document The type of the actual document content being stored.
 * @property content The actual document content to be stored.
 * @property documentType An optional classification or type identifier for the document (e.g., "article", "report").
 * @property source An optional identifier for the document's origin or source system.
 * @property tags A list of tags or labels associated with the document for categorization and filtering.
 * @property metadata An optional map of arbitrary key-value pairs for storing additional structured metadata.
 *
 * Note: If other storage implementations find this useful, this class may be moved to a more general location
 * in the codebase (e.g., `ai.koog.rag.base`) to promote reusability across different storage backends.
 */
@Serializable
public data class DocumentWithMetadata<Document>(
    val content: Document,
    val documentType: String? = null,
    val source: String? = null,
    val tags: List<String> = emptyList(),
    val metadata: Map<String, JsonElement>? = null
)

/**
 * A persistent storage implementation of the [VectorStorage] interface backed by PostgreSQL and the pgvector extension.
 *
 * This class enables storing, retrieving, and managing documents with their associated vector embeddings in a PostgreSQL table.
 * Documents are wrapped in [DocumentWithMetadata] to capture additional metadata fields (such as document type, source, tags, and arbitrary metadata),
 * which are persisted in dedicated columns alongside the serialized document content and embedding.
 *
 * Embedding vectors are stored either as dense float vectors or as binary (bit) vectors, depending on [vectorType].
 * Efficient approximate nearest neighbor (ANN) search is supported by creating HNSW indexes over the embedding column,
 * using only those [distanceOperators] that are valid for the selected [vectorType].
 *
 * At initialization, only valid HNSW indexes are created for the specified operator-type combinations.
 * Using an unsupported operator for a given vector type will throw an error at initialization.
 *
 * Serialization and deserialization of document content is handled using a [KSerializer] and a [Json] instance, ensuring type safety and integration
 * with Kotlin's kotlinx.serialization framework. This enables you to store any `@Serializable` data class as the document content.
 *
 * @param Document The type of document content to be stored (e.g., a serializable data class).
 *
 * @constructor
 * @param connectionProvider A function that provides a fresh [Connection] to the PostgreSQL database. Each operation gets a new connection from this.
 * @param vectorDimension The number of dimensions for the embedding vectors stored in the table. This must match the dimensionality of your vectors.
 * @param serializer A [KSerializer] used to (de)serialize [Document] content to and from JSON.
 * @param json The [Json] instance to use for (de)serialization. Defaults to the standard [Json] configuration, but can be customized.
 * @param tableName The name of the PostgreSQL table where documents and vector embeddings will be stored. Defaults to "vector_store" if not specified.
 * @param distanceOperators The set of distance/similarity operators to support for ANN queries (see [VectorDistanceOperator]). Only valid indexes are created.
 * @param vectorType The underlying type of vector for storage and indexing: [VectorStorageType.FLOAT_VECTOR] for standard embeddings, [VectorStorageType.BIT_VECTOR] for binary vectors.
 */
public class PgVectorStorage<Document> private constructor(
    private val connectionProvider: () -> Connection,
    private val vectorDimension: Int,
    private val serializer: KSerializer<Document>,
    private val json: Json = Json,
    private val tableName: String = "vector_store",
    private val distanceOperators: Set<VectorDistanceOperator>,
    private val vectorType: VectorStorageType
) : VectorStorage<DocumentWithMetadata<Document>> {

    /** Factory functions for construction of PgVectorStorage instances, as well as helper functions and const values */
    public companion object {

        /**
         * Creates a FLOAT_VECTOR PgVectorStorage for standard dense (float) embeddings; sets up L2 as the default operator.
         */
        public fun <Document> floatVectorStorage(
            connectionProvider: () -> Connection,
            vectorDimension: Int,
            serializer: KSerializer<Document>,
            json: Json = Json,
            tableName: String = "vector_store",
            distanceOperators: Set<VectorDistanceOperator> = setOf(VectorDistanceOperator.L2)
        ): PgVectorStorage<Document> {
            validateOperatorCombo(VectorStorageType.FLOAT_VECTOR, distanceOperators)
            return PgVectorStorage(
                connectionProvider = connectionProvider,
                vectorDimension = vectorDimension,
                serializer = serializer,
                json = json,
                tableName = tableName,
                distanceOperators = distanceOperators,
                vectorType = VectorStorageType.FLOAT_VECTOR
            )
        }

        /**
         * Creates a BIT_VECTOR PgVectorStorage for binary (0/1) embeddings, using HAMMING as the default distance operator.
         */
        public fun <Document> bitVectorStorage(
            connectionProvider: () -> Connection,
            vectorDimension: Int,
            serializer: KSerializer<Document>,
            json: Json = Json,
            tableName: String = "bit_vector_store",
            distanceOperators: Set<VectorDistanceOperator> = setOf(VectorDistanceOperator.HAMMING)
        ): PgVectorStorage<Document> {
            validateOperatorCombo(VectorStorageType.BIT_VECTOR, distanceOperators)
            return PgVectorStorage(
                connectionProvider = connectionProvider,
                vectorDimension = vectorDimension,
                serializer = serializer,
                json = json,
                tableName = tableName,
                distanceOperators = distanceOperators,
                vectorType = VectorStorageType.BIT_VECTOR
            )
        }

        private fun validateOperatorCombo(
            vectorType: VectorStorageType,
            distanceOperators: Set<VectorDistanceOperator>
        ) {
            when (vectorType) {
                VectorStorageType.FLOAT_VECTOR -> {
                    val forbidden = distanceOperators.filter {
                        it in setOf(VectorDistanceOperator.HAMMING, VectorDistanceOperator.JACCARD)
                    }
                    require(forbidden.isEmpty()) {
                        "Operators $forbidden are not valid for FLOAT_VECTOR. Only L2, L1, COSINE, and DOT_PRODUCT are allowed."
                    }
                }
                VectorStorageType.BIT_VECTOR -> {
                    val forbidden = distanceOperators.filter {
                        it !in setOf(VectorDistanceOperator.HAMMING, VectorDistanceOperator.JACCARD)
                    }
                    require(forbidden.isEmpty()) {
                        "Operators $forbidden are not valid for BIT_VECTOR. Only HAMMING and JACCARD are allowed."
                    }
                }
            }
        }
    }

    /** The set of distance/similarity operators supported by this storage instance. */
    public val getDistanceOperators: Set<VectorDistanceOperator>
        get() = distanceOperators

    /** The type of vector (float or bit) used by this storage instance. */
    public val getVectorType: VectorStorageType
        get() = vectorType

    init {
        val dimensionLimit = when (vectorType) {
            VectorStorageType.FLOAT_VECTOR -> 2000
            VectorStorageType.BIT_VECTOR -> 64000
            // Add future vector types here
            // VectorStorageType.HALFVEC -> 4000
            // VectorStorageType.SPARSEVEC -> 1000 // non-zero elements, special case
        }
        require(vectorDimension <= dimensionLimit) {
            "PgVector HNSW index supports at most $dimensionLimit dimensions for type $vectorType (requested $vectorDimension); see https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw"
        }
        // Idempotent Table and Index Setup
        connectionProvider().use { connection ->
            PGvector.registerTypes(connection)
            connection.createStatement().use { statement ->
                statement.executeUpdate("CREATE EXTENSION IF NOT EXISTS vector")
                statement.executeUpdate("""
                CREATE TABLE IF NOT EXISTS $tableName (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding ${
                    when (vectorType) {
                        VectorStorageType.FLOAT_VECTOR -> "vector($vectorDimension) NOT NULL"
                        VectorStorageType.BIT_VECTOR -> "bit($vectorDimension) NOT NULL"
                    }
                },
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    document_type TEXT,
                    source TEXT,
                    tags TEXT[],
                    metadata JSONB
                )
            """.trimIndent())
                when (vectorType) {
                    VectorStorageType.FLOAT_VECTOR -> {
                        distanceOperators.forEach { op ->
                            when (op) {
                                VectorDistanceOperator.L2 -> statement.executeUpdate(
                                    "CREATE INDEX IF NOT EXISTS idx_embedding_l2 ON $tableName USING hnsw (embedding vector_l2_ops)"
                                )
                                VectorDistanceOperator.L1 -> statement.executeUpdate(
                                    "CREATE INDEX IF NOT EXISTS idx_embedding_l1 ON $tableName USING hnsw (embedding vector_l1_ops)"
                                )
                                VectorDistanceOperator.COSINE -> statement.executeUpdate(
                                    "CREATE INDEX IF NOT EXISTS idx_embedding_cosine ON $tableName USING hnsw (embedding vector_cosine_ops)"
                                )
                                VectorDistanceOperator.DOT_PRODUCT -> statement.executeUpdate(
                                    "CREATE INDEX IF NOT EXISTS idx_embedding_ip ON $tableName USING hnsw (embedding vector_ip_ops)"
                                )
                                // HAMMING and JACCARD are NOT valid for FLOAT_VECTOR
                                else -> error("Operator $op is not supported for FLOAT_VECTOR columns.")
                            }
                        }
                    }
                    VectorStorageType.BIT_VECTOR -> {
                        distanceOperators.forEach { op ->
                            when (op) {
                                VectorDistanceOperator.HAMMING -> statement.executeUpdate(
                                    "CREATE INDEX IF NOT EXISTS idx_embedding_hamming ON $tableName USING hnsw (embedding bit_hamming_ops)"
                                )
                                VectorDistanceOperator.JACCARD -> statement.executeUpdate(
                                    "CREATE INDEX IF NOT EXISTS idx_embedding_jaccard ON $tableName USING hnsw (embedding bit_jaccard_ops)"
                                )
                                // L2, L1, COSINE, DOT_PRODUCT are NOT valid for BIT_VECTOR
                                else -> error("Operator $op is not supported for BIT_VECTOR columns.")
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Stores a document with metadata and its associated embedding vector in the database.
     * If a document with the same ID already exists, it is updated (upsert).
     *
     * @param document The document with metadata to store.
     * @param data The vector embedding associated with the document.
     * @return The UUID (as string) assigned to the stored document.
     */
    override suspend fun store(document: DocumentWithMetadata<Document>, data: Vector): String {
        require(data.values.size == vectorDimension) {
            "Vector size ${data.values.size} does not match configured dimension $vectorDimension for storage"
        }
        if (vectorType == VectorStorageType.BIT_VECTOR) {
            require(data.values.all { it == 0.0 || it == 1.0 }) {
                "BIT_VECTOR storage only supports vector values 0.0 or 1.0"
            }
        }
        val id = UUID.randomUUID()
        val content = json.encodeToString(serializer, document.content)
        val vector = when (vectorType) {
            VectorStorageType.FLOAT_VECTOR -> PGvector(data.toFloatArray())
            VectorStorageType.BIT_VECTOR -> {
                val bitString = data.values.joinToString(separator = "") { if (it != 0.0) "1" else "0" }
                PGbit(bitString)
            }
        }
        val metadataJson = document.metadata?.let { Json.encodeToString(it) }
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val statement = conn.prepareStatement(
                """
                INSERT INTO $tableName (id, content, embedding, document_type, source, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?::jsonb)
                ON CONFLICT (id) DO UPDATE SET 
                    content = EXCLUDED.content, 
                    embedding = EXCLUDED.embedding,
                    document_type = EXCLUDED.document_type,
                    source = EXCLUDED.source,
                    tags = EXCLUDED.tags,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """.trimIndent()
            )
            statement.setObject(1, id)
            statement.setString(2, content)
            statement.setObject(3, vector)
            statement.setString(4, document.documentType)
            statement.setString(5, document.source)
            statement.setArray(6, conn.createArrayOf("TEXT", document.tags.toTypedArray()))
            statement.setString(7, metadataJson)
            statement.executeUpdate()
            statement.close()
        }
        return id.toString()
    }

    /**
     * Deletes a document and its embedding from the database by document ID.
     *
     * @param documentId The UUID string of the document to delete.
     * @return True if a document was deleted, false if no document with the given ID existed.
     */
    override suspend fun delete(documentId: String): Boolean {
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val stmt = conn.prepareStatement(
                "DELETE FROM $tableName WHERE id = ?"
            )
            stmt.setObject(1, UUID.fromString(documentId))
            val affected = stmt.executeUpdate()
            stmt.close()
            return affected > 0
        }
    }

    /**
     * Reads and deserializes a stored document with its metadata by document ID.
     *
     * @param documentId The UUID string of the document to read.
     * @return The document with metadata if found, or null if not found.
     */
    override suspend fun read(documentId: String): DocumentWithMetadata<Document>? =
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val statement = conn.prepareStatement(
                "SELECT content, document_type, source, tags, metadata FROM $tableName WHERE id = ?"
            )
            statement.setObject(1, UUID.fromString(documentId))
            val resultSet = statement.executeQuery()
            val documentWithMetadata = if (resultSet.next())
                documentWithMetadataFromResultSet(resultSet)
            else null
            resultSet.close(); statement.close()
            documentWithMetadata
        }

    /**
     * Reads and returns the vector embedding for the given document ID.
     *
     * @param documentId The UUID string of the document.
     * @return The [Vector] embedding if found, or null if no such document.
     */
    override suspend fun getPayload(documentId: String): Vector? =
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val statement = conn.prepareStatement("SELECT embedding FROM $tableName WHERE id = ?")
            statement.setObject(1, UUID.fromString(documentId))
            val resultSet = statement.executeQuery()
            val vector =
                if (resultSet.next()) {
                    val embeddingObj = resultSet.getObject("embedding")
                    when (vectorType) {
                        VectorStorageType.FLOAT_VECTOR -> {
                            val floatArray: FloatArray = when (embeddingObj) {
                                is PGvector -> embeddingObj.toArray()
                                is PGobject -> PGvector(embeddingObj.value).toArray()
                                else -> error("Unknown embedding type: ${embeddingObj?.javaClass}")
                            }
                            Vector(floatArray.map { it.toDouble() })
                        }
                        VectorStorageType.BIT_VECTOR -> {
                            val bitString = (embeddingObj as PGobject).value
                            requireNotNull(bitString) { "BIT_VECTOR column returned null value (expected a 0/1 string)" }
                            Vector(bitString.map { if (it == '1') 1.0 else 0.0 })
                        }
                    }
                } else null
            resultSet.close(); statement.close()
            vector
        }

    /**
     * Reads both the deserialized document with metadata and its associated vector embedding by document ID.
     *
     * @param documentId The UUID string of the document.
     * @return [DocumentWithPayload] containing the document with metadata and vector, or null if not found.
     */
    override suspend fun readWithPayload(documentId: String): DocumentWithPayload<DocumentWithMetadata<Document>, Vector>? =
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val statement = conn.prepareStatement(
                "SELECT content, embedding, document_type, source, tags, metadata FROM $tableName WHERE id = ?"
            )
            statement.setObject(1, UUID.fromString(documentId))
            val resultSet = statement.executeQuery()
            val documentWithPayload =
                if (resultSet.next()) {
                    val embeddingObj = resultSet.getObject("embedding")
                    val vector = when (vectorType) {
                        VectorStorageType.FLOAT_VECTOR -> when (embeddingObj) {
                            is PGvector -> Vector(embeddingObj.toArray().map { it.toDouble() })
                            is PGobject -> Vector(PGvector(embeddingObj.value).toArray().map { it.toDouble() })
                            else -> error("Unknown embedding type for FLOAT_VECTOR: ${embeddingObj?.javaClass}")
                        }
                        VectorStorageType.BIT_VECTOR -> {
                            val bitString = (embeddingObj as PGobject).value
                            requireNotNull(bitString) { "BIT_VECTOR column returned null value (expected a 0/1 string)" }
                            Vector(bitString.map { if (it == '1') 1.0 else 0.0 })
                        }
                    }
                    val documentWithMetadata = documentWithMetadataFromResultSet(resultSet)
                    DocumentWithPayload(documentWithMetadata, vector)
                } else null
            resultSet.close(); statement.close()
            documentWithPayload
        }

    /**
     * Returns a cold [Flow] emitting all deserialized documents with metadata in the table.
     *
     * @return A [Flow] emitting every stored document with metadata.
     */
    override fun allDocuments(): Flow<DocumentWithMetadata<Document>> = flow {
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val statement = conn.createStatement()
            val resultSet = statement.executeQuery("SELECT content, document_type, source, tags, metadata FROM $tableName")
            while (resultSet.next()) {
                emit(documentWithMetadataFromResultSet(resultSet))
            }
            resultSet.close(); statement.close()
        }
    }

    /**
     * Returns a cold [Flow] emitting all deserialized documents with metadata and their associated embeddings.
     *
     * @return A [Flow] emitting [DocumentWithPayload] objects for each document stored.
     */
    override fun allDocumentsWithPayload(): Flow<DocumentWithPayload<DocumentWithMetadata<Document>, Vector>> = flow {
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val statement = conn.createStatement()
            val resultSet = statement.executeQuery(
                "SELECT content, embedding, document_type, source, tags, metadata FROM $tableName"
            )
            while (resultSet.next()) {
                val embeddingObj = resultSet.getObject("embedding")
                val vector = when (vectorType) {
                    VectorStorageType.FLOAT_VECTOR -> when (embeddingObj) {
                        is PGvector -> Vector(embeddingObj.toArray().map { it.toDouble() })
                        is org.postgresql.util.PGobject -> Vector(PGvector(embeddingObj.value).toArray().map { it.toDouble() })
                        else -> error("Unknown embedding type for FLOAT_VECTOR: ${embeddingObj?.javaClass}")
                    }
                    VectorStorageType.BIT_VECTOR -> {
                        val bitString = (embeddingObj as org.postgresql.util.PGobject).value
                        requireNotNull(bitString) { "BIT_VECTOR column returned null value (expected a 0/1 string)" }
                        Vector(bitString.map { if (it == '1') 1.0 else 0.0 })
                    }
                }
                val documentWithMetadata = documentWithMetadataFromResultSet(resultSet)
                emit(DocumentWithPayload(documentWithMetadata, vector))
            }
            resultSet.close(); statement.close()
        }
    }

    private fun documentWithMetadataFromResultSet(rs: ResultSet): DocumentWithMetadata<Document> {
        val content = json.decodeFromString(serializer, rs.getString("content"))
        val documentType = rs.getString("document_type")
        val source = rs.getString("source")
        val tagsArray = rs.getArray("tags")
        val tags = (tagsArray?.array as? Array<*>)?.mapNotNull { it as? String } ?: emptyList()
        val metadataJson = rs.getString("metadata")
        val metadata = metadataJson?.let { json.decodeFromString<Map<String, JsonElement>>(it) }
        return DocumentWithMetadata(
            content = content,
            documentType = documentType,
            source = source,
            tags = tags,
            metadata = metadata
        )
    }

    /**
     * Searches for the top-K most similar documents to a query vector, using a configurable
     * vector distance or similarity operator from the PostgreSQL pgvector extension.
     *
     * This method streams the results as a [Flow], allowing for efficient, lazy processing.
     * Each emitted value is a pair of the stored document (with metadata) and its distance (or similarity) score
     * as computed by the selected operator.
     *
     * The query leverages database-native vector search for fast approximate nearest neighbor (ANN) lookups.
     * The distance or similarity used for ranking is determined by the [operator] parameter
     * (see [VectorDistanceOperator]), supporting operators such as L2 (Euclidean), cosine distance, dot product, etc.
     *
     * **Operator Restriction**: Only operators present in the set passed to the constructor
     * of this [PgVectorStorage] instance are allowed. Attempting to use an operator that
     * was not configured for this instance will result in an [IllegalArgumentException].
     * This ensures that only supported operators with corresponding database indexes are used.
     *
     * Note: The underlying JDBC operations are blocking; collect this [Flow] on a dispatcher such as
     * [kotlinx.coroutines.Dispatchers.IO] to avoid blocking the main thread.
     *
     * @param queryVector The embedding vector to compare against stored documents.
     * @param topK The number of nearest neighbors (most similar results) to return.
     * @param operator The pgvector distance or similarity operator to use for search and ranking. Must be one of the operators configured for this storage instance.
     * @return A [Flow] emitting pairs of [DocumentWithMetadata] and distance (or similarity) score,
     *         ordered exactly as returned by the database. For most distance metrics (L2, L1, COSINE, HAMMING, JACCARD),
     *         the first element is the most similar (smallest distance). For DOT_PRODUCT, the scores are negative inner products,
     *         so the first element is the most negative (lowest dot product); the actual most similar (largest original dot product) will be last.
     *         To get "most similar first" for DOT_PRODUCT results, consume the entire flow and select the last element.
     * @throws IllegalArgumentException if [operator] is not supported by this storage instance, or is COSINE with a zero vector.
     */
    public fun topKSimilarDocumentsWithOperator(
        queryVector: Vector,
        topK: Int = 5,
        operator: VectorDistanceOperator
    ): Flow<Pair<DocumentWithMetadata<Document>, Double>> = flow {
        require(operator in distanceOperators) {
            "Operator $operator is not supported by this PgVectorStorage instance. Supported: $distanceOperators"
        }
        require(!(operator == VectorDistanceOperator.COSINE && queryVector.values.all { it == 0.0 })) {
            "COSINE operator does not support zero query vector (undefined cosine similarity)."
        }
        connectionProvider().use { conn ->
            PGvector.registerTypes(conn)
            val vecStr = when (vectorType) {
                VectorStorageType.FLOAT_VECTOR ->
                    queryVector.values.joinToString(prefix = "[", postfix = "]", separator = ",")
                VectorStorageType.BIT_VECTOR ->
                    queryVector.values.joinToString(separator = "") { if (it != 0.0) "1" else "0" }
            }
            val sql = """
            SELECT *, embedding ${operator.sql} '$vecStr' AS distance
            FROM $tableName
            ORDER BY embedding ${operator.sql} '$vecStr'
            LIMIT $topK
        """.trimIndent()
            val stmt = conn.prepareStatement(sql)
            val rs = stmt.executeQuery()
            while (rs.next()) {
                val docWithMeta = documentWithMetadataFromResultSet(rs)
                val distance = rs.getDouble("distance")
                emit(docWithMeta to distance)
            }
            rs.close(); stmt.close()
        }
    }

    /**
     * Streams the top-K most similar documents, outputting a normalized similarity score for each,
     * where similarity is computed as 1 / (1 + distance). This means higher scores indicate greater similarity,
     * with a perfect match producing a score of 1 and less similar results approaching 0.
     *
     * This normalization assumes the distance is a true non-negative metric (e.g., L2/Euclidean, cosine distance, L1, Hamming, Jaccard).
     * For these operators, lower distance always means more similar, and the normalized score reflects that.
     *
     * **Operator Restriction**: Only operators present in the set passed to the constructor
     * of this [PgVectorStorage] instance are allowed. Attempting to use an operator that
     * was not configured for this instance will result in an [IllegalArgumentException].
     * This ensures that only supported operators with corresponding database indexes are used.
     *
     * This function will throw if used with the DOT_PRODUCT operator. Dot product distances can be negative or very large, and lower
     * values mean more similar, so the normalization (1 / (1 + distance)) does not produce meaningful similarity scores in this case.
     *
     * This function will also throw if used with the COSINE operator and the provided query vector is all zeros,
     * as cosine similarity is undefined for zero vectors.
     *
     * The method returns a [Flow] so callers can process results lazily in a coroutine pipeline.
     * Note: The underlying JDBC operations are blocking; collect this flow on a dispatcher such as
     * [kotlinx.coroutines.Dispatchers.IO] for best practice.
     *
     * @param queryVector The embedding vector to compare against stored documents.
     * @param topK The number of results to return.
     * @param operator The vector distance or similarity operator to use for scoring and ranking. Must be one of the operators configured for this storage instance.
     * @return A [Flow] emitting pairs of [DocumentWithMetadata] and normalized similarity score.
     * @throws IllegalArgumentException if the operator is unsupported, DOT_PRODUCT, or COSINE with a zero vector.
     */
    public fun topKSimilarDocumentsWithNormalizedSimilarity(
        queryVector: Vector,
        topK: Int = 5,
        operator: VectorDistanceOperator
    ): Flow<Pair<DocumentWithMetadata<Document>, Double>> = flow {
        require(operator in distanceOperators) {
            "Operator $operator is not supported by this PgVectorStorage instance. Supported: $distanceOperators"
        }
        require(operator != VectorDistanceOperator.DOT_PRODUCT) {
            "Normalized similarity does not support DOT_PRODUCT operator. " +
                "Dot product scores may be negative and do not represent a true distance; " +
                "use only with metric distance operators (L2, L1, cosine, Hamming, Jaccard)."
        }
        require(!(operator == VectorDistanceOperator.COSINE && queryVector.values.all { it == 0.0 })) {
            "Normalized similarity does not support COSINE operator with a zero vector. " +
                "Cosine similarity is undefined for all-zero vectors."
        }
        topKSimilarDocumentsWithOperator(
            queryVector = queryVector,
            topK = topK,
            operator = operator
        ).collect { (doc, distance) ->
            val similarity = 1.0 / (1.0 + distance)
            emit(doc to similarity)
        }
    }
}

/**
 * Represents the supported vector distance and similarity operators
 * available in the PostgreSQL pgvector extension for nearest-neighbor search.
 * Each operator is represented by its SQL syntax.
 *
 * Documentation: https://github.com/pgvector/pgvector?tab=readme-ov-file#querying
 *
 * - [L2]           - Euclidean (L2) distance: `<->`
 * - [DOT_PRODUCT]  - Negative inner product: `<#>`
 * - [COSINE]       - Cosine distance: `<=>`
 * - [L1]           - Manhattan (L1) distance: `<+>`
 * - [HAMMING]      - Hamming distance for binary vectors: `<~>`
 * - [JACCARD]      - Jaccard distance for binary vectors: `<%>`
 *
 * @property sql The SQL string for the operator`.
 */
public enum class VectorDistanceOperator(public val sql: String) {
    L2("<->"),
    DOT_PRODUCT("<#>"),
    COSINE("<=>"),
    L1("<+>"),
    HAMMING("<~>"),
    JACCARD("<%>")
}

/**
 * Indicates the underlying type of vector used for storage and querying.
 * - FLOAT_VECTOR: regular dense float vectors (common for embeddings).
 * - BIT_VECTOR: binary (0/1) vectors, for use with Hamming/Jaccard in pgvector.
 */
public enum class VectorStorageType {
    FLOAT_VECTOR,
    BIT_VECTOR
}

/**
 * A document storage implementation that pairs your documents with vector embeddings,
 * stored in Postgres using pgvector for scalable similarity search and retrieval.
 *
 * Whenever you store a document, this class will automatically embed your document using the provided [embedder]
 * before saving both the document and its vector representation.
 *
 * It also provides efficient ranking and retrieval: to find the most relevant documents for a query string,
 * the query is first embedded into a vector (again using [embedder]), and then the database searches for and ranks documents by similarity.
 *
 * - [topK] and [operator] (distance metric) are exposed as open properties so you can override them per subclass.
 *   Set these to configure the number of results and the similarity function used.
 *
 * **Important:** The value of [operator] must be one of the operators configured for the underlying [PgVectorStorage].
 * If not, a runtime error will be thrown when ranking documents. Always ensure your custom `operator` override matches
 * an operator that is present in the distanceOperators set when initializing the storage.
 *
 * @param Document The user-defined type of the document being stored and ranked.
 * @param embedder Converts input documents and queries (as strings) into vector embeddings, for both storage and retrieval.
 * @param storage  The PgVectorStorage instance for persistent vector/document management.
 *
 * This class is intended for scenarios where you want to leverage database-native approximate nearest neighbor (ANN) search,
 * using your preferred embedder pipeline for both storing documents and querying for relevance.
 */
public open class PgVectorRankedDocumentStorage<Document>(
    embedder: DocumentEmbedder<DocumentWithMetadata<Document>>,
    private val storage: PgVectorStorage<Document>
) : EmbeddingBasedDocumentStorage<DocumentWithMetadata<Document>>(embedder, storage) {

    /**
     * The number of top results to return for rankDocuments().
     */
    protected open val topK: Int get() = 5

    /**
     * The similarity/distance operator used in rankDocuments().
     *
     * **Warning:** This value must be present in the storage's allowed distanceOperators set,
     * or a runtime error will occur when searching.
     */
    protected open val operator: VectorDistanceOperator
        get() = VectorDistanceOperator.L2

    override fun rankDocuments(query: String): Flow<RankedDocument<DocumentWithMetadata<Document>>> = flow {
        if (operator !in storage.getDistanceOperators) {
            error("Configured operator $operator is not supported by this PgVectorStorage instance. Supported: ${storage.getDistanceOperators}")
        }
        val queryVector = embedder.embed(query)
        storage.topKSimilarDocumentsWithNormalizedSimilarity(
            queryVector = queryVector,
            topK = topK,
            operator = operator
        ).collect { (doc, similarity) ->
            emit(RankedDocument(document = doc, similarity = similarity))
        }
    }
}
