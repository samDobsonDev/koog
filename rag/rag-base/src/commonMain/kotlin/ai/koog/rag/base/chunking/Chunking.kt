package ai.koog.rag.base.chunking

import ai.koog.rag.base.files.DocumentProvider
import ai.koog.rag.base.files.TextRange
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * A chunk of a document, tagged with range, its text, and optional metadata.
 */
@Serializable
public data class DocumentChunk<Document>(
    val parent: Document,
    val range: TextRange,
    val text: String
)

/**
 * Generic interface for splitting a document into chunks.
 */
public interface DocumentChunker<Document> {
    /**
     * Split the given document into zero or more chunks.
     */
    public suspend fun chunk(document: Document): List<DocumentChunk<Document>>

    /**
     * Human-readable description of this chunking strategy.
     */
    public val description: String
}

/**
 * A generic chunker for custom document types, which extracts the document's string representation
 * using a [DocumentProvider] and then delegates the actual chunking to a string-based [DocumentChunker].
 *
 * This allows you to seamlessly apply standard string chunking strategies (like ParagraphChunker, SlidingWindowChunker, etc.)
 * to custom or rich document types (e.g. data classes, files, PDFs, parsed objects, etc.)—as long as you have a [DocumentProvider]
 * implementation to extract plain text from your document type.
 *
 * This is especially useful for cases where your corpus is not simple strings (e.g., it's files, network objects, or rich data structures),
 * but you still want to leverage your standard chunkers for pipelines like RAG, semantic search, etc.
 *
 * @param Document The custom document type to chunk.
 * @param Path The path or key type, as required by your [DocumentProvider] (often may be the same as [Document]).
 * @property documentProvider The provider used to extract textual content from the document.
 * @property delegate The string-based chunker used to split the extracted text into chunks.
 */
public class DocumentProviderDelegatingChunker<Document, Path>(
    private val documentProvider: DocumentProvider<Path, Document>,
    private val delegate: DocumentChunker<String>
) : DocumentChunker<Document> {

    override val description: String =
        "Chunks using DocumentProvider and delegates to: ${delegate.description}"

    override suspend fun chunk(document: Document): List<DocumentChunk<Document>> {
        val text = documentProvider.text(document).toString()
        return delegate.chunk(text).map { chunk ->
            DocumentChunk(
                parent = document,
                range = chunk.range,
                text = chunk.text
            )
        }
    }
}

/**
 * Example: Paragraph chunker for plain text documents.
 * Splits on two or more newlines.
 */
public class ParagraphChunker : DocumentChunker<String> {
    override val description: String = "Chunks text into paragraphs using blank lines as boundaries."

    override suspend fun chunk(document: String): List<DocumentChunk<String>> {
        val result = mutableListOf<DocumentChunk<String>>()
        var offset = 0
        val regex = Regex("""(\r?\n){2,}""")
        regex.findAll(document).forEach { match ->
            val end = match.range.first
            val chunkText = document.substring(offset, end)
            if (chunkText.isNotBlank()) {
                result.add(
                    DocumentChunk(
                        parent = document,
                        range = TextRange(offset, end),
                        text = chunkText.trim()
                    )
                )
            }
            offset = match.range.last + 1
        }
        // Last chunk (if any)
        if (offset < document.length) {
            val chunkText = document.substring(offset)
            if (chunkText.isNotBlank()) {
                result.add(
                    DocumentChunk(
                        parent = document,
                        range = TextRange(offset, document.length),
                        text = chunkText.trim()
                    )
                )
            }
        }
        return result
    }
}

/**
 * Example: Sliding window chunker for text, with tokens=approximate by char count.
 * @param windowSize Number of characters per chunk
 * @param overlap Number of characters that overlap between chunks
 */
public class SlidingWindowChunker(
    private val windowSize: Int,
    private val overlap: Int = 0
) : DocumentChunker<String> {
    override val description: String get() =
        "Sliding window chunker (window=$windowSize, overlap=$overlap)"

    override suspend fun chunk(document: String): List<DocumentChunk<String>> {
        val result = mutableListOf<DocumentChunk<String>>()
        var start = 0
        while (start < document.length) {
            val end = (start + windowSize).coerceAtMost(document.length)
            val chunkText = document.substring(start, end)
            result.add(
                DocumentChunk(
                    parent = document,
                    range = TextRange(start, end),
                    text = chunkText
                )
            )
            if (end == document.length) break
            start += (windowSize - overlap).coerceAtLeast(1)
        }
        return result
    }
}
