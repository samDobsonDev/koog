package ai.koog.integration.tests.rag

import ai.koog.embeddings.base.Vector
import ai.koog.embeddings.local.LLMEmbedder
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.integration.tests.utils.TestUtils
import ai.koog.rag.vector.DocumentEmbedder
import ai.koog.rag.vector.DocumentWithMetadata
import ai.koog.rag.vector.PgVectorRankedDocumentStorage
import ai.koog.rag.vector.PgVectorStorage
import ai.koog.rag.vector.VectorDistanceOperator
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import org.junit.jupiter.api.*
import org.testcontainers.containers.PostgreSQLContainer
import org.testcontainers.utility.DockerImageName
import java.sql.Connection
import java.sql.DriverManager
import kotlin.test.assertTrue

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PgVectorRankedDocumentStorageIntegrationTest {

    companion object {
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
            DriverManager.getConnection(postgres.jdbcUrl, postgres.username, postgres.password)
        }
    }

    @AfterAll
    fun stopContainer() {
        postgres.stop()
    }

    @Serializable data class MyDoc(val title: String, val body: String)

    class MyDocEmbedder(val base: LLMEmbedder) : DocumentEmbedder<DocumentWithMetadata<MyDoc>> {
        override suspend fun embed(document: DocumentWithMetadata<MyDoc>): Vector =
            base.embed("${document.content.title} ${document.content.body}")
        override suspend fun embed(text: String): Vector = base.embed(text)
        override fun diff(embedding1: Vector, b: Vector): Double = base.diff(embedding1, b)
    }

    @Test
    fun integration_storesEmbedsAndRanksRealDocumentsWithOpenAIEmbedder() = runBlocking {

        // Create the embedder
        val apiKey = TestUtils.readTestOpenAIKeyFromEnv()
        val embedder = LLMEmbedder(
            client = OpenAILLMClient(apiKey = apiKey),
            model = OpenAIModels.Embeddings.TextEmbedding3Small
        )
        val docEmbedder = MyDocEmbedder(embedder)

        val storage = PgVectorStorage.floatVectorStorage(
            connectionProvider = connProvider,
            vectorDimension = 1536,
            serializer = MyDoc.serializer(),
            tableName = "integration_ranked_docs",
            distanceOperators = setOf(VectorDistanceOperator.COSINE)
        )

        val rankedStorage = object : PgVectorRankedDocumentStorage<MyDoc>(docEmbedder, storage) {
            override val topK: Int get() = 3
            override val operator: VectorDistanceOperator get() = VectorDistanceOperator.COSINE
        }

        // Store several example documents (embedding is done automatically by rankedStorage)
        val docs = listOf(
            DocumentWithMetadata(content = MyDoc("First", "The quick brown fox jumps over the lazy dog.")),
            DocumentWithMetadata(content = MyDoc("Second", "A guide to making fluffy pancakes at home.")),
            DocumentWithMetadata(content = MyDoc("Third", "The history of artificial intelligence and its future.")),
        )
        docs.forEach { rankedStorage.store(it) }

        // Now rank for a search query
        val query = "Tips for perfect breakfast"
        val results = rankedStorage.rankDocuments(query).toList()
        println("Results for query: '$query'")
        results.forEach {
            println("Title: ${it.document.content.title} | Sim: ${"%.3f".format(it.similarity)} | Body: ${it.document.content.body}")
        }

        // Ensure that we get a meaningful ranking (the pancake article scores highest)
        assertTrue(results.isNotEmpty())
        assertTrue(results.first().document.content.title == "Second" || results.first().document.content.body.contains("pancake", ignoreCase = true))
    }
}
