package ai.koog.rag.base.chunking

import ai.koog.rag.base.files.DocumentProvider
import kotlin.test.*
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.Serializable

class ChunkingTest {

    @Test
    fun `paragraph chunker splits plain text by paragraphs`() = runTest {
        val doc = "First paragraph.\n\nSecond one!\n\nThird."
        val chunker = ParagraphChunker()
        val chunks = chunker.chunk(doc)
        assertEquals(3, chunks.size)
        assertEquals("First paragraph.", chunks[0].text)
        assertEquals("Second one!", chunks[1].text)
        assertEquals("Third.", chunks[2].text)
        assertEquals(doc, chunks[0].parent)
        assertTrue(chunks[1].range.start < chunks[1].range.endExclusive)
    }

    @Serializable
    data class MyDoc(val title: String, val body: String)

    @Test
    fun `document provider chunker splits custom doc by body paragraphs`() = runTest {
        val doc = MyDoc(
            title = "Header",
            body = "Intro.\n\nDetail paragraph.\n\nConclusion."
        )
        // A DocumentProvider that extracts the 'body' as the text
        val docProvider = object : DocumentProvider<MyDoc, MyDoc> {
            override suspend fun document(path: MyDoc) = path
            override suspend fun text(document: MyDoc) = document.body
        }
        val customChunker = DocumentProviderDelegatingChunker(
            documentProvider = docProvider,
            delegate = ParagraphChunker()
        )
        val chunks = customChunker.chunk(doc)
        assertEquals(3, chunks.size)
        assertEquals("Intro.", chunks[0].text)
        assertEquals(doc, chunks[1].parent)
        assertTrue(chunks[2].text.contains("Conclusion"))
    }

    @Test
    fun `sliding window chunker creates overlapping chunks`() = runTest {
        val doc = "abcdefghij"
        val chunker = SlidingWindowChunker(windowSize = 5, overlap = 2)
        val chunks = chunker.chunk(doc).map { it.text }
        assertEquals(listOf("abcde", "defgh", "ghij"), chunks)
    }
}
