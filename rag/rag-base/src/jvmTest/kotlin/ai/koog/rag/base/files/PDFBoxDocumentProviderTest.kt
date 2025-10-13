package ai.koog.rag.base.files

import ai.koog.rag.base.chunking.*
import kotlinx.coroutines.test.runTest
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.PDPageContentStream
import org.apache.pdfbox.pdmodel.font.PDType1Font
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.io.File

class PDFBoxDocumentProviderTest {

    @Test
    fun `can extract chunks from a generated PDF using DocumentProviderDelegatingChunker`() = runTest {
        val tempFile = File.createTempFile("test-sample", ".pdf")
        PDDocument().use { doc ->
            val page = PDPage()
            doc.addPage(page)
            PDPageContentStream(doc, page).use { cs ->
                cs.beginText()
                cs.setFont(PDType1Font.HELVETICA, 12f)
                cs.newLineAtOffset(50f, 700f)
                val lines = listOf("First paragraph.", "Second paragraph!", "Third line.")
                for (line in lines) {
                    cs.showText(line)
                    cs.newLineAtOffset(0f, -20f)
                }
                cs.endText()
            }
            doc.save(tempFile)
        }
        val chunker = DocumentProviderDelegatingChunker(
            documentProvider = PDFBoxDocumentProvider,
            delegate = ParagraphChunker()
        )
        val pdDoc = PDFBoxDocumentProvider.document(tempFile)
        requireNotNull(pdDoc) { "Failed to load PDF." }
        val chunks = chunker.chunk(pdDoc)
        assertTrue(chunks.isNotEmpty(), "Should extract at least one chunk from PDF")
        assertTrue(chunks.all { it.text.isNotBlank() }, "No chunks should be blank")
        assertTrue(chunks.all { it.parent == pdDoc }, "Parent reference should be the original PDDocument")
        chunks.forEach { println("Chunk: ${it.text}") }
        tempFile.delete()
    }
}
