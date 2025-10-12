# PgVectorStorage (PostgreSQL Vector Storage with pgvector)

Koog’s `PgVectorStorage` provides robust, persistent storage and efficient similarity search for document embeddings, leveraging the [pgvector PostgreSQL extension](https://github.com/pgvector/pgvector) for high-performance vector indexing and querying.

This storage backend allows you to efficiently store document vectors, metadata, and retrieve the most relevant documents using a variety of vector similarity/distance metrics, supporting both dense (float) embeddings and binary (bit) vectors.

## Key Features

- **Persistent vector storage**: Embeddings are stored in a PostgreSQL table (with the flexible pgvector extension.
- **Flexible schema**: Store both document contents and rich metadata (type tag, source, custom tags, arbitrary metadata map).
- **High-performance retrieval**: Utilizes PostgreSQL’s HNSW ANN (approximate nearest neighbor) indexes for fast top-K similarity search.
- **Supports multiple similarity/distance operators**: Choose from L2, L1, cosine, dot product (for floats), or Hamming/Jaccard (for bit vectors).
- **Automatic table and index creation**: Instantiates and manages the database table schema (and indexes) as needed.
- **Kotlin serialization integration**: Use any serializable data class for your document content; JSON (de)serialization is fully automatic.

## When to Use PgVectorStorage

- For **scalable, persistent semantic search** over large document sets.
- When you need **rich metadata filtering and auditability**.
- When you want to leverage native ANN search and indexing for embeddings.

## How It Works

1. **Initialize with a JDBC Connection Provider**
    - You provide a connection (or pool) to a PostgreSQL instance with pgvector installed.
2. **Specify vector type and dimensionality**
    - Supports both float and bit vectors for dense or binary embeddings.
3. **Document Ingestion**
    - Store documents and their embeddings, including optional tags and metadata.
4. **Efficient Retrieval**
    - Top-K similarity search with flexible operator, leveraging database-native ANN search.

## Using `PgVectorStorage` in your RAG pipeline

### Setup

```kotlin
@Serializable
data class MyDoc(val title: String, val body: String)

// Create a JDBC connection provider (could also come from a pool manager)
val connProvider = { DriverManager.getConnection("jdbc:postgresql://host:port/db", "user", "pass") }
```

### Create a PgVectorStorage Instance

For **dense float vector storage** (embeddings as floats, default operator: L2/Euclidean distance):

```kotlin
val floatStorage = PgVectorStorage.floatVectorStorage(
    connectionProvider = connProvider,
    vectorDimension = 768,
    serializer = MyDoc.serializer(),
    tableName = "vector_storage_table"
)
```

For **bit vector storage** (embeddings as binary 0/1 vectors, default operator: Hamming distance):

```kotlin
val bitStorage = PgVectorStorage.bitVectorStorage(
    connectionProvider = connProvider,
    vectorDimension = 256,
    serializer = MyDoc.serializer(),
    tableName = "bit_vector_table"
)
```

> **Limitation:** For HNSW indexing, pgvector limits *float* vector columns to 2,000 dimensions max and *bit* vector columns to 64,000 dimensions max.  
> Attempting to create storage with a higher dimension for either type will fail.  
> See: https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw

### **Options when creating a `floatVectorStorage()` or `bitVectorStorage()` instance**

Both storage constructors share several configurable options, but some are specialized for float or bit vector use cases.

**Common parameters:**
- `connectionProvider`: Lambda providing JDBC connections (required).
- `vectorDimension`: The expected dimension (length) of all vectors (required).
- `serializer`: The `KSerializer` for your document class (required).
- `json`: (optional) A custom Kotlinx `Json` instance for serialization/deserialization.
- `tableName`: (optional) The PostgreSQL table name (default is `"vector_store"` for float or `"bit_vector_store"` for bit).

**For `floatVectorStorage()`:**
- `distanceOperators`: (optional) A set of distance/similarity operators to enable for ANN searches and index creation.
    - **Default:** `setOf(VectorDistanceOperator.L2)` (Euclidean distance)
    - **Available for float:**
        - `VectorDistanceOperator.L2` (Euclidean)
        - `VectorDistanceOperator.L1` (Manhattan)
        - `VectorDistanceOperator.COSINE` (Cosine similarity)
        - `VectorDistanceOperator.DOT_PRODUCT` (Negative inner product)
    - *You can supply more than one operator if you want to support/benchmark different types of search/query on the same data.*

**For `bitVectorStorage()`:**
- `distanceOperators`: (optional) A set of operators to support ANN queries and indexes.
    - **Default:** `setOf(VectorDistanceOperator.HAMMING)`
    - **Available for bit:**
        - `VectorDistanceOperator.HAMMING`
        - `VectorDistanceOperator.JACCARD`

**Example:**
```kotlin
val storage = PgVectorStorage.floatVectorStorage(
    connectionProvider = connProvider,
    vectorDimension = 384,
    serializer = MyDoc.serializer(),
    distanceOperators = setOf(
        VectorDistanceOperator.L2,
        VectorDistanceOperator.COSINE
    ),
    tableName = "vector_store"
)
```

This stores vectors in the `"vector_store"` table and enables the use of both L2 and cosine queries with full ANN search speed.

> Attempting to use an invalid operator for the storage type (e.g., Hamming for float vectors) will throw an error on initialization.

### Storing Documents

```kotlin
// Create a DocumentWithMetadata object to store in the database.
// Pass your custom document data class as the 'content' parameter.
val document = DocumentWithMetadata(
    content = MyDoc(
        title = "Hello",
        body = "World!"
    ),
    documentType = "article",
    source = "blog",
    tags = listOf("example"),
    metadata = mapOf("author" to JsonPrimitive("koog"))
)

// Generate an embedding for the document using an embedder
val embedding = embedder.embed(document)

// Store the document and it's associated vector representation in the database
val id = floatStorage.store(document, embedding)
```

### Performing Similarity Search

```kotlin
val queryEmbedding = Vector(listOf(0.1, 0.2, 0.3, /* ... */ ))

// Collect the top-3 most similar results as a list.
// Since the function returns a Flow, use .toList() inside a coroutine to eagerly retrieve all results.
val topKResults = floatStorage
    .topKSimilarDocumentsWithOperator(queryEmbedding, topK = 3, operator = VectorDistanceOperator.L2)
    .toList()

for ((doc, distance) in topKResults) {
    println("Document: ${doc.content}, Distance: $distance")
}
```

### Normalized Similarity Search

To get a score in the range [0, 1], where 1.0 is a perfect match and 0.0 means no similarity (useful for ranking and UI display), use the `topKSimilarDocumentsWithNormalizedSimilarity` function:

```kotlin
// Normalized similarity is calculated as 1.0 / (1.0 + distance).
// Supported for L2, L1, COSINE, HAMMING, and JACCARD operators.
val topKSimilarities = floatStorage
    .topKSimilarDocumentsWithNormalizedSimilarity(
        queryVector = queryEmbedding,
        topK = 3,
        operator = VectorDistanceOperator.L2
    )
    .toList()

for ((doc, similarity) in topKSimilarities) {
    println("Document: ${doc.content}, Normalized similarity: $similarity")
}
```

- Use with supported operators **only**: L2, L1, COSINE, HAMMING, or JACCARD (not DOT_PRODUCT).
- For COSINE, make sure your query vector is not all zeros (cosine similarity with a zero vector is mathematically undefined).
- The higher the normalized similarity, the closer the match.

## All-in-One: Ranked Document Storage with Database-Native ANN Search

For most use cases, you want a complete “RAG ready” ranked document storage that combines embedding, storage, and efficient search in a single class, delegating all ranking and retrieval to the database engine.

Koog’s `PgVectorRankedDocumentStorage` provides exactly this: it wires together embedding, metadata tracking, storage, and retrieval via PostgreSQL + pgvector’s powerful similarity search.

### What is `PgVectorRankedDocumentStorage`?

- It’s an implementation of a ranked document storage for RAG, which:
    - Automatically embeds query strings into vectors using your chosen embedder.
    - Stores documents—complete with full metadata—alongside their embeddings in the database.
    - Performs high-performance top-K similarity search within Postgres using ANN (HNSW) vector indexes, rather than computing similarities in application memory.
- Supports all document types that can be embedded via your chosen embedder.
- You can configure the number of results and operator for similarity search by overriding the `topK` and `operator` properties in a subclass.

### Typical Usage

```kotlin
@Serializable
data class MyDoc(val title: String, val body: String)

// Define an embedder
val embedder = ...

// Create your PgVectorStorage
val storage = PgVectorStorage.floatVectorStorage(
    connectionProvider = connProvider,
    vectorDimension = 768,
    serializer = MyDoc.serializer(),
    distanceOperators = setOf(VectorDistanceOperator.COSINE)
)

// Instantiate the ranked storage
val rankedDocumentStorage = object : PgVectorRankedDocumentStorage<MyDoc>(embedder, storage) {
    // Optionally override these properties in a subclass/object to customize ranking behavior:
    override val topK: Int get() = 10
    override val operator: VectorDistanceOperator get() = VectorDistanceOperator.COSINE
}

// Store a document (same as with PgVectorStorage)
// Pass your custom document data as the 'content' parameter.
val document = DocumentWithMetadata(
    content = MyDoc(title = "Hello", body = "World!"),
    documentType = "article"
)
val vector = embedder.embed(document)
rankedDocumentStorage.store(document, vector)

// Find the most relevant documents for a user query
val query = "hello"
val topDocs = rankedDocumentStorage.rankDocuments(query).toList()
for (doc in topDocs) {
    println("Ranked doc: ${doc.document.content}, similarity: ${doc.similarity}")
}
```

> **Warning:** The `operator` property must match one of the operators configured on your `PgVectorStorage` instance, or a runtime error will occur.

---

### When to Use

Use `PgVectorRankedDocumentStorage` when you want:
- A single high-level API for both storing and semantically ranking documents via PostgreSQL ANN search.
- To seamlessly integrate your embedder and your pgvector-powered document store.

## Additional Functionality

To see the full set of operations available for storing, retrieving, deleting, and ranking documents—including batch retrieval, payload inspection, and more—refer to the core [Ranked Document Storage documentation](ranked-document-storage.md) for usage examples and details.

`PgVectorStorage` implements the VectorStorage interface and supports all documented functions.

## Supported Operators

Depending on your vector type:

- **FLOAT_VECTOR**: L2 (Euclidean), L1 (Manhattan), COSINE, DOT_PRODUCT (negative)
- **BIT_VECTOR**: HAMMING, JACCARD

See the [Vector Operations Reference](pgvector-operations.md) for detailed semantics on each.

## Operator Selection and Indexing

- PgVectorStorage will **validate that you only configure valid operators for each vector type** and will only create applicable HNSW or GiST indexes in PostgreSQL.
- If you attempt to search with an operator not configured for this instance, a clear error is thrown.

## Table and Index Initialization

- On creation, `PgVectorStorage` will:
    - Ensure the required `vector` extension is installed in your database.
    - Automatically create the required table (if missing), with columns for content, embedding, type, tags, etc.
    - Create HNSW/ANN indexes for all configured operators on the embedding column.
- **All schema changes are idempotent** (safe to run at app startup).

## Caveats / Tips

- For **COSINE** operator, the query vector **must not be all zeros** (cosine similarity is undefined).
- For **DOT_PRODUCT**, note that the search is on the negative inner product; most similar is the last entry in a top-K ASC list.
- Make sure your embedding dimensionality matches the vectorDimension parameter, or the database will reject inserts.
